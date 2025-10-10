# translate.py
from functools import lru_cache
from typing import List, Optional, Union
import warnings
import torch
from packaging import version

# Marian (preferred)
from transformers import MarianMTModel, MarianTokenizer

# Optional M2M fallback
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def _device(auto: Optional[str] = None) -> str:
    if auto:
        return auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_ge_26() -> bool:
    try:
        return version.parse(torch.__version__) >= version.parse("2.6.0")
    except Exception:
        return False


# Check SentencePiece (needed by Marian & M2M tokenizers)
try:
    import sentencepiece  # noqa: F401
    _SPM_OK = True
except Exception:
    _SPM_OK = False


class HFTranslator:
    """
    MarianMT FR<->EN translator.

    - Uses safetensors to avoid torch.load (and its CVE gate on old torch).
    - Loads to 'cuda' if available, else CPU.
    """
    def __init__(
        self,
        fr_en_model: str = "Helsinki-NLP/opus-mt-fr-en",
        en_fr_model: str = "Helsinki-NLP/opus-mt-en-fr",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        if not _SPM_OK:
            raise RuntimeError("SentencePiece is required. Please `pip install sentencepiece`.")

        self.device = _device(device)
        self.max_new_tokens = max_new_tokens

        # Tokenizers
        self.tok_fr_en = MarianTokenizer.from_pretrained(fr_en_model)
        self.tok_en_fr = MarianTokenizer.from_pretrained(en_fr_model)

        # Models — force safetensors; eager load; move to device; eval
        self.mod_fr_en = MarianMTModel.from_pretrained(
            fr_en_model,
            use_safetensors=True,      # avoid torch.load on .bin
            low_cpu_mem_usage=False,   # eager, no meta tensors
        ).to(self.device).eval()

        self.mod_en_fr = MarianMTModel.from_pretrained(
            en_fr_model,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        ).to(self.device).eval()

    @torch.inference_mode()
    def translate(self, text: Union[str, List[str]], src: str, tgt: str) -> Union[str, List[str]]:
        if src == tgt:
            return text
        if isinstance(text, str):
            return self._one(text, src, tgt)
        return [self._one(t, src, tgt) for t in text]

    def _one(self, text: str, src: str, tgt: str) -> str:
        s = (text or "").strip()
        if not s:
            return text

        # Pick the right direction
        if src.lower().startswith("fr") and tgt.lower().startswith("en"):
            tok, mod = self.tok_fr_en, self.mod_fr_en
        elif src.lower().startswith("en") and tgt.lower().startswith("fr"):
            tok, mod = self.tok_en_fr, self.mod_en_fr
        else:
            # Not supported by this 2-direction Marian setup
            return text

        enc = tok(s, return_tensors="pt", truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = mod.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        return tok.batch_decode(out, skip_special_tokens=True)[0]


class M2MTranslator:
    """
    M2M100 FR<->EN translator with safetensors.
    Safe CPU-only load to avoid meta tensor paths and device-map quirks.
    """
    def __init__(
        self,
        model_name: str = "facebook/m2m100_418M",
        device: Optional[str] = None,  # ignored; forced CPU
        max_new_tokens: int = 512,
    ):
        if not _SPM_OK:
            raise RuntimeError("SentencePiece is required. Please `pip install sentencepiece`.")

        self.device = "cpu"  # force CPU
        self.max_new_tokens = max_new_tokens

        self.tok = M2M100Tokenizer.from_pretrained(model_name, use_fast=False)

        # Load fully on CPU, no device_map, no meta init, no .to(...)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
            low_cpu_mem_usage=False,    # make real tensors immediately
            torch_dtype=torch.float32,  # safe for CPU/Windows
        ).eval()

    @torch.inference_mode()
    def translate(self, text: Union[str, List[str]], src: str, tgt: str) -> Union[str, List[str]]:
        if src == tgt:
            return text
        if isinstance(text, str):
            return self._one(text, src, tgt)
        return [self._one(t, src, tgt) for t in text]

    def _one(self, text: str, src: str, tgt: str) -> str:
        s = (text or "").strip()
        if not s:
            return text
        src = (src or "fr")[:2].lower()
        tgt = (tgt or "en")[:2].lower()
        self.tok.src_lang = src
        inputs = self.tok(s, return_tensors="pt", truncation=True)  # CPU tensors
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tok.get_lang_id(tgt),
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        return self.tok.batch_decode(generated, skip_special_tokens=True)[0]


@lru_cache(maxsize=1)
def get_translator(
    provider: str = "marian",               # "marian" (preferred) or "m2m"
    fr_en_model: str = "Helsinki-NLP/opus-mt-fr-en",
    en_fr_model: str = "Helsinki-NLP/opus-mt-en-fr",
    model_name: str = "facebook/m2m100_418M",
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    cache_bust: str = "v2",                # bump to refresh the cache
):
    prov = (provider or "marian").lower()

    if prov == "m2m":
        try:
            return M2MTranslator(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
        except Exception as e:
            warnings.warn(f"M2MTranslator failed to load ({e}). Falling back to Marian.", RuntimeWarning)
            # fall through to Marian

    # Default to Marian (safetensors; works on torch<2.6 too)
    try:
        return HFTranslator(
            fr_en_model=fr_en_model,
            en_fr_model=en_fr_model,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        # If Marian fails for any reason, try M2M as a last resort
        warnings.warn(f"HFTranslator (Marian) failed to load ({e}). Trying M2M.", RuntimeWarning)
        return M2MTranslator(model_name=model_name, device=device, max_new_tokens=max_new_tokens)


def translate_text(text: Union[str, List[str]], src: str, tgt: str, translator=None) -> Union[str, List[str]]:
    tr = translator or get_translator()
    return tr.translate(text, src, tgt)

