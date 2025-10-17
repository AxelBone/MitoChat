"""Streamlit app with multiple messages session
V5:
* Using all models locally
* Adding router decision to call RAG or internal knowledge
* Adding identifiers checking to force RAG to check Cninvar or Mitocarta
* Adding pretty sources visualization
"""



import os
import json
import base64
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from translate_v1 import get_translator, translate_text
import io
from PIL import Image
from streamlit.runtime.media_file_storage import MediaFileStorageError


# ---- Global safe image patch ----
if not getattr(st, "_image_patched", False):
    _orig_st_image = st.image

    HASHY_IMG_RE = re.compile(r"^[0-9a-f]{32,64}\.(?:png|jpg|jpeg|gif)$", re.I)

    def _st_image_safe(obj=None, *args, **kwargs):
        try:
            # Accept bytes/bytearray/BytesIO/PIL/numpy as-is
            import io, numpy as _np
            from PIL import Image as _PILImage

            if isinstance(obj, (bytes, bytearray, io.BytesIO, _PILImage.Image, _np.ndarray)):
                return _orig_st_image(obj, *args, **kwargs)

            # Strings: only display if it's a real file or an http(s) URL
            if isinstance(obj, str):
                if HASHY_IMG_RE.match(obj):
                    st.caption("Image not available.")
                    return
                if obj.startswith(("http://", "https://")) or os.path.exists(obj):
                    return _orig_st_image(obj, *args, **kwargs)
                st.caption("Image not available.")
                return

            # Fallback: try original, but guard
            return _orig_st_image(obj, *args, **kwargs)
        except MediaFileStorageError:
            st.caption("Image not available.")
        except Exception:
            st.caption("Image not available.")

    st.image = _st_image_safe
    st._image_patched = True
# ---- end patch ----



# Device helpers
def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# Optional reranker
try:
    from sentence_transformers import CrossEncoder
    _RERANK_OK = True
except Exception:
    _RERANK_OK = False

# Optional token trimming
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

# llama.cpp (in-process LLM, recommended default)
try:
    from llama_cpp import Llama
    _LLAMACPP_OK = True
except Exception:
    _LLAMACPP_OK = False

# Optional Transformers provider (in-process; requires GPU or patience on CPU)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    _HF_OK = True
except Exception:
    _HF_OK = False


# ----------------------------
# Config / data loading
# ----------------------------
def read_config() -> Dict[str, Any]:
    cfg_path = os.getenv("CONFIG", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def load_index(index_dir: str):
    index_path = os.path.join(index_dir, "index.faiss")
    ds_path = os.path.join(index_dir, "docstore.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    if not os.path.exists(index_path) or not os.path.exists(ds_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing index/docstore/meta under {index_dir}")

    index = faiss.read_index(index_path)
    if not isinstance(index, faiss.IndexIDMap2):
        index = faiss.IndexIDMap2(index)

    docstore = []
    with open(ds_path, "r", encoding="utf-8") as f:
        for line in f:
            docstore.append(json.loads(line))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, docstore, meta


# @st.cache_resource(show_spinner=False)
# def load_embedder(model_name: str) -> SentenceTransformer:
#     return SentenceTransformer(model_name)
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> SentenceTransformer:
    # Using GPU (or MPS) if available; falls back to CPU
    return SentenceTransformer(model_name, device=pick_device())


@st.cache_resource(show_spinner=False)
def load_subindex(index_dir: str, name: str) -> Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]:
    ipath = os.path.join(index_dir, f"index_{name}.faiss")
    dpath = os.path.join(index_dir, f"docstore_{name}.jsonl")
    if not (os.path.exists(ipath) and os.path.exists(dpath)):
        return None, None
    ix = faiss.read_index(ipath)
    if not isinstance(ix, faiss.IndexIDMap2):
        ix = faiss.IndexIDMap2(ix)
    ds = []
    with open(dpath, "r", encoding="utf-8") as f:
        for line in f:
            ds.append(json.loads(line))
    return ix, ds

@st.cache_resource(show_spinner=False)
def load_id_maps(index_dir: str) -> Dict[str, Any]:
    p = os.path.join(index_dir, "id_maps.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"variant_id_to_idx": {}, "rsid_to_idx": {}, "gene_symbol_to_idxs": {}}


@st.cache_resource(show_spinner=False)
def load_rewrite_prompts(lang: str = "fr", base_dir: str = "prompts") -> dict:
    path = os.path.join(base_dir, f"rewrite_{lang}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # minimal validation
    for k in ("system", "fewshots", "output_tags"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {path}")
    if "open" not in data["output_tags"] or "close" not in data["output_tags"]:
        raise ValueError(f"Missing output_tags.open/close in {path}")
    return data


@st.cache_resource(show_spinner=False)
def load_router_prompt(lang: str = "en", base_dir: str = "prompts") -> dict:
    path = os.path.join(base_dir, f"router_{lang}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Router prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for k in ("system", "user_template", "output_format"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in router prompt {path}")
    return data


# --- NER loader (put near other @st.cache_resource loaders) ---
@st.cache_resource(show_spinner=False)
def load_ner(model_name: str = "en_core_web_sm"):
    """
    Tries sciSpaCy first if present, otherwise falls back to spaCy small.
    Install suggestions:
      pip install spacy
      python -m spacy download en_core_web_sm
    For biomedical: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    """
    try:
        import spacy
        return spacy.load(model_name)
    except Exception:
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except Exception:
            return None  # app will gracefully fallback



def rewrite_query_llm_from_yaml(
    chat_history: list[dict],
    user_input: str,
    provider: str,
    gen_cfg: dict,
    prompts: dict,
) -> tuple[str, dict]:
    sys_prompt = prompts["system"].strip()
    fewshots = prompts["fewshots"].strip()
    max_turns = int(prompts.get("max_turns", 4))
    open_tag = prompts["output_tags"]["open"]
    close_tag = prompts["output_tags"]["close"]
    tag_re = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL)

    recent = chat_history[-max_turns*2:] if chat_history else []
    lines = []
    for m in recent:
        if m.get("role") in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content','').strip()}")
    history_text = "\n".join(lines)

    user_prompt = f"""{fewshots}

Recent history:
{history_text}

Current question:
Utilisateur: {user_input}

Rewrite the current question so that it is self-contained while strictly following the constraints.
Remember: no invention. Keep the original language. Preserve genes/variants/acronyms exactly as they are.
Output (strictly): {open_tag}…{close_tag}
""".strip()

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    debug = {"provider": provider, "messages": messages, "raw": None, "error": None}

    try:
        if provider == "llama_cpp":
            llm = get_llama_cpp_model(
                model_path=gen_cfg.get("model_path"),
                num_ctx=int(gen_cfg.get("num_ctx", 2048)),
                n_gpu_layers=int(gen_cfg.get("n_gpu_layers", 0)),
                n_threads=int(gen_cfg.get("n_threads", 8)),
                n_batch=int(gen_cfg.get("n_batch", 256)),
                verbose=False,
            )
            if llm is None:
                return user_input, {**debug, "error": "llama.cpp model not available"}
            out = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=120)
            text = out["choices"][0]["message"]["content"] or ""
            debug["raw"] = text
        else:
            answer = generate_with_hf(messages, {
                "model": gen_cfg.get("model"),
                "quantization": gen_cfg.get("quantization", "4bit"),
                "device_map": gen_cfg.get("device_map", "auto"),
                "temperature": 0.0,
                "max_tokens": 120,
                "trust_remote_code": gen_cfg.get("trust_remote_code", False),
            })
            text = answer or ""
            debug["raw"] = text

        m = tag_re.search(text.strip())
        if m:
            rewritten = m.group(1).strip()
            if rewritten:
                return rewritten, debug

        # fallback: if model returned a single non-empty line (no tags)
        fallback = text.strip()
        if fallback and fallback != user_input:
            return fallback, debug
        return user_input, debug

    except Exception as e:
        debug["error"] = str(e)
        return user_input, debug


def render_router_user(router_cfg: dict, query: str, hits: List[Dict[str, Any]]) -> str:
    # Format passages as: [i] • score • source • text
    parts = []
    for i, h in enumerate(hits, 1):
        score = h.get("rerank_score", h.get("score", 0.0))
        src = pretty_citation(h.get("metadata", {}) or {})
        txt = (h.get("text") or "").strip()
        # keep each snippet short for routing
        snippet = txt[:800]
        parts.append(f"[{i}] • {score:.3f} • {src}\n{snippet}")
    passages = "\n\n".join(parts) if parts else "(none)"
    return router_cfg["user_template"].format(query=query, passages=passages).strip()


def call_router_llm(
    router_prompt: dict,
    query: str,
    hits: List[Dict[str, Any]],
    provider: str,
    gen_cfg: Dict[str, Any],
) -> dict:
    system = router_prompt["system"].strip()
    user = render_router_user(router_prompt, query, hits)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    raw = None
    try:
        if provider == "llama_cpp":
            raw = generate_with_llama_cpp(messages, {
                "model_path": gen_cfg.get("model_path"),
                "num_ctx": int(gen_cfg.get("num_ctx", 2048)),
                "n_gpu_layers": int(gen_cfg.get("n_gpu_layers", 0)),
                "n_threads": int(gen_cfg.get("n_threads", 8)),
                "n_batch": int(gen_cfg.get("n_batch", 256)),
                "temperature": 0.0,
                "max_tokens": 128,
            })
        else:
            raw = generate_with_hf(messages, {
                "model": gen_cfg.get("model"),
                "quantization": gen_cfg.get("quantization", "4bit"),
                "device_map": gen_cfg.get("device_map", "auto"),
                "temperature": 0.0,
                "max_tokens": 128,
                "trust_remote_code": gen_cfg.get("trust_remote_code", False),
            })
    except Exception:
        raw = None

    # Parse a single-line JSON object anywhere in the text
    out = {"mode": "RAG" if hits else "LLM", "reason": "fallback", "use_ids": []}
    if not raw:
        return out

    m = re.search(r"\{.*\}", raw.replace("\n", " ").strip())
    if not m:
        return out
    try:
        j = json.loads(m.group(0))
        mode = str(j.get("mode", "")).upper()
        if mode not in {"RAG", "LLM", "HYBRID"}:
            return out
        use_ids = j.get("use_ids") or []
        # keep only valid ids
        use_ids = [int(i) for i in use_ids if isinstance(i, int) and 1 <= i <= len(hits)]
        return {"mode": mode, "reason": j.get("reason", ""), "use_ids": use_ids}
    except Exception:
        return out



rewrite_prompts = load_rewrite_prompts(lang="en", base_dir="../prompts")
router_prompt = load_router_prompt(lang="en", base_dir="../prompts")


def embed_query(query: str, embedder: SentenceTransformer) -> np.ndarray:
    emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


# ----------------------------
# Retrieval utilities
# ----------------------------
def pretty_citation(md: Dict[str, Any]) -> str:
    src = (md or {}).get("source")
    if src == "genereviews":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id") or "Document"
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "clinvar":
        vid = md.get("variant_id")
        gs = md.get("gene_symbol")
        return f"ClinVar: {vid or 'variant'} in {gs or 'gene'}"
    if src == "mitocarta":
        return f"Mitocarta gene: {md.get('symbol')}"
    # Legacy fallback keys
    if src == "pdf":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id") or "Document"
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "json_variants":
        return f"Variant {md.get('variant_id')} in {md.get('gene_symbol')}"
    if src == "json_genes":
        return f"Gene {md.get('symbol')}"
    # Fallback to source_file basename
    sf = md.get("source_file")
    return Path(sf).name if sf else "Source"



GUIDANCE_KEYWORDS_DEFAULT = [
    "frequency","surveillance","monitor","monitoring","follow-up","follow up",
    "management","treatment","avoid","contraindicated","recommend","recommended",
    "ekg","ecg","echocardiogram","holter","blood pressure","annually","every 6"
]

def classify_guidance_query(q: str, retr_cfg: Dict[str, Any]) -> bool:
    if not retr_cfg.get("enforce_pdf_for_guidance", True):
        return False
    kws = retr_cfg.get("guidance_keywords") or GUIDANCE_KEYWORDS_DEFAULT
    ql = q.lower()
    return any(k.lower() in ql for k in kws)

def _normalize_boosts(source_boosts: Dict[str, float]) -> Dict[str, float]:
    boosts = dict(source_boosts or {})
    # Map legacy keys -> new keys
    if "pdf" in boosts and "genereviews" not in boosts:
        boosts["genereviews"] = boosts["pdf"]
    if "json_variants" in boosts and "clinvar" not in boosts:
        boosts["clinvar"] = boosts["json_variants"]
    if "json_genes" in boosts and "mitocarta" not in boosts:
        boosts["mitocarta"] = boosts["json_genes"]
    return boosts

def apply_source_boosts(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], prefer_genereviews: bool) -> None:
    boosts = _normalize_boosts(retr_cfg.get("source_boosts") or {})
    if prefer_genereviews:
        boosts["genereviews"] = boosts.get("genereviews", 0.0) + 0.30
        boosts["clinvar"] = boosts.get("clinvar", 0.0) - 0.10
        boosts["mitocarta"] = boosts.get("mitocarta", 0.0) - 0.05
    for h in hits:
        base = float(h.get("rerank_score", h.get("score", 0.0)))
        src = (h.get("metadata") or {}).get("source")
        h["_adj_score"] = base + float(boosts.get(src, 0.0))

def lexical_signal_genereviews(query: str, text: str, retr_cfg: Dict[str, Any]) -> float:
    if not retr_cfg.get("hybrid_lexical", True):
        return 0.0
    w = float(retr_cfg.get("lexical_weight_pdf", 0.02))  # keep key name for compat
    tl = text.lower()
    count = 0
    for t in GUIDANCE_KEYWORDS_DEFAULT:
        count += len(re.findall(r"\b" + re.escape(t.lower()) + r"\b", tl))
    return w * count

def add_lexical_boosts_for_genereviews(hits: List[Dict[str, Any]], query: str, retr_cfg: Dict[str, Any]) -> None:
    for h in hits:
        md = h.get("metadata") or {}
        if md.get("source") == "genereviews":
            h["_adj_score"] = h.get("_adj_score", float(h.get("rerank_score", h.get("score", 0.0)))) \
                              + lexical_signal_genereviews(query, h["text"], retr_cfg)

def _cap_values(retr_cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    caps = retr_cfg.get("per_source_cap") or {}
    gr_min = int(caps.get("genereviews_min", caps.get("pdf_min", 3)))
    clinvar_max = int(caps.get("clinvar_max", caps.get("json_variants_max", 2)))
    mitocarta_max = int(caps.get("mitocarta_max", caps.get("json_genes_max", 2)))
    return gr_min, clinvar_max, mitocarta_max

def rebalance_by_source(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], final_k: int) -> List[Dict[str, Any]]:
    gr_min, clinvar_max, mitocarta_max = _cap_values(retr_cfg)
    hits_sorted = sorted(
        hits, key=lambda x: x.get("_adj_score", x.get("rerank_score", x.get("score", 0.0))), reverse=True
    )
    gr = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "genereviews"]
    cl = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "clinvar"]
    mt = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "mitocarta"]
    other = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") not in {"genereviews","clinvar","mitocarta"}]

    result, used = [], set()
    # 1) ensure minimum GeneReviews
    for h in gr[:min(gr_min, len(gr))]:
        if h["idx"] not in used:
            result.append(h); used.add(h["idx"])
    # 2) limited JSONs
    def take_cap(pool, cap):
        src = (pool[0].get("metadata") or {}).get("source") if pool else None
        added = 0
        for h in pool:
            if added >= cap:
                break
            if h["idx"] in used:
                continue
            result.append(h); used.add(h["idx"]); added += 1
    take_cap(cl, clinvar_max)
    take_cap(mt, mitocarta_max)
    # 3) fill remaining
    def take(pool):
        for h in pool:
            if len(result) >= final_k:
                break
            if h["idx"] in used:
                continue
            result.append(h); used.add(h["idx"])
    if len(result) < final_k: take(gr)
    if len(result) < final_k: take(other)
    if len(result) < final_k: take(cl)
    if len(result) < final_k: take(mt)
    return result[:final_k]


# Extract identifiers inside sentences
VARIANT_ID_RE = re.compile(r"\b(?:(?:chr)?([0-9]{1,2}|X|Y|MT)):(\d+):([ACGT]):([ACGT])\b", re.I)
RSID_RE = re.compile(r"\brs\d+\b", re.I)

def detect_identifiers(query: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in VARIANT_ID_RE.finditer(query):
        chrom = m.group(1).upper(); pos = m.group(2)
        ref = m.group(3).upper(); alt = m.group(4).upper()
        out.append(("variant_id", f"{chrom}:{pos}:{ref}:{alt}"))
    for m in RSID_RE.finditer(query):
        out.append(("rsid", m.group(0).lower()))
    m = re.search(r"\bgene\s+([A-Z0-9\-]{2,10})\b", query)
    if m:
        out.append(("gene_symbol", m.group(1).upper()))
    elif query.strip().isupper() and 2 <= len(query.strip()) <= 10:
        out.append(("gene_symbol", query.strip().upper()))
    return out

def structured_lookup_first(query: str, id_maps: Dict[str, Any], docstore: List[Dict[str, Any]], limit_per_key: int = 3) -> List[Dict[str, Any]]:
    idents = detect_identifiers(query)
    if not idents:
        return []
    hits: List[Dict[str, Any]] = []
    seen_idx: set[int] = set()
    for which, key in idents:
        if which == "variant_id":
            idxs = id_maps.get("variant_id_to_idx", {}).get(key, [])
        elif which == "rsid":
            m = id_maps.get("rsid_to_idx", {})
            idxs = m.get(key, []) + m.get(key.replace("rs", ""), [])
        elif which == "gene_symbol":
            idxs = id_maps.get("gene_symbol_to_idxs", {}).get(key, [])
        else:
            idxs = []
        for i in idxs[:limit_per_key]:
            if i in seen_idx:
                continue
            rec = docstore[i]
            hits.append({
                "score": 1.0,
                "idx": int(i),
                "text": rec["text"],
                "metadata": rec.get("metadata", {}),
                "_adj_score": 999.0,
            })
            seen_idx.add(i)
    return hits

def _hit_key(h: Dict[str, Any]) -> Tuple[Any, ...]:
    md = h.get("metadata", {}) or {}
    src = md.get("source")
    return (src, md.get("variant_id") or md.get("symbol") or md.get("doc_id"), md.get("page"), md.get("section"))


def search(
    query: str,
    index: faiss.IndexIDMap2,
    docstore: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int,
    enable_reranker: bool,
    rerank_top_k: int,
    reranker_model: str,
    retr_cfg: Optional[Dict[str, Any]] = None,
    id_maps: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    retr_cfg = retr_cfg or {}

    # 0) Structured fast-path
    struct_hits = structured_lookup_first(query, id_maps or {}, docstore, limit_per_key=3)

    # 1) Dense vector search
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qemb, top_k)
    I = I[0]; D = D[0]

    dense_hits = []
    for score, idx in zip(D, I):
        if idx == -1: continue
        rec = docstore[idx]
        dense_hits.append({"score": float(score), "idx": int(idx), "text": rec["text"], "metadata": rec.get("metadata", {})})

    # 2) Optional reranker
    if enable_reranker and _RERANK_OK and rerank_top_k and dense_hits:
        try:
            ce = CrossEncoder(reranker_model, device=pick_device())
            rr = ce.predict([(query, h["text"]) for h in dense_hits], batch_size=128, show_progress_bar=False)
            for i, s in enumerate(rr): dense_hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in dense_hits: h["rerank_score"] = h["score"]
    else:
        for h in dense_hits: h["rerank_score"] = h["score"]

    # 3) Merge struct + dense, dedupe
    hits = []
    seen = set()
    for h in struct_hits + dense_hits[: (rerank_top_k or len(dense_hits))]:
        k = _hit_key(h)
        if k in seen: continue
        hits.append(h); seen.add(k)

    # 4) Boosts + lexical
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)

    # 5) Per-source quotas
    final_k = int(retr_cfg.get("final_k", rerank_top_k or len(hits)))
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits


def search_multi_stage(
    query: str,
    embedder: SentenceTransformer,
    per_source_indices: Dict[str, Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]],
    per_source_k: Dict[str, int],
    final_k: int,
    enable_reranker: bool,
    reranker_model: str,
    retr_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def search_ix(ix, ds, k):
        if not ix or not ds or k <= 0: return []
        D, I = ix.search(qemb, k); D, I = D[0], I[0]
        out = []
        for score, idx in zip(D, I):
            if idx == -1: continue
            rec = ds[idx]
            out.append({"score": float(score), "idx": int(idx), "text": rec["text"], "metadata": rec.get("metadata", {})})
        return out

    # normalize keys
    key_map = {"genereviews": "genereviews", "pdf": "genereviews", "clinvar": "clinvar", "json_variants": "clinvar", "mitocarta": "mitocarta", "json_genes": "mitocarta"}
    hits = []
    for k_raw, k in key_map.items():
        if k_raw in per_source_k:
            ix, ds = per_source_indices.get(k, (None, None))
            hits.extend(search_ix(ix, ds, int(per_source_k[k_raw])))

    if not hits: return []

    # Rerank
    if enable_reranker and _RERANK_OK:
        try:
            ce = CrossEncoder(reranker_model, device=pick_device())
            rr = ce.predict([(query, h["text"]) for h in hits], batch_size=128, show_progress_bar=False)
            for i, s in enumerate(rr): hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in hits: h["rerank_score"] = h["score"]
    else:
        for h in hits: h["rerank_score"] = h["score"]

    # Boosts + lexical + quotas
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits

# ----------------------------
# Chunks retrieval utility
# ----------------------------
def resolve_full_chunk(hit: Dict[str, Any], docstore: Any) -> Tuple[str, Any]:
    """
    Returns (kind, content)
    kind in {"text", "json"}
    content is a string for text, or a dict/list for json.

    NOTE: We intentionally DO NOT use hit["idx"] to index into the global docstore,
    because hits can come from per-source subindices where idx != global index.
    """
    md = (hit.get("metadata") or {})

    # 0) Prefer full content already stored in metadata
    if md.get("section_text"):
        return "text", str(md["section_text"])
    if md.get("json_entry"):
        return "json", md["json_entry"]
    if md.get("raw_json"):
        try:
            return "json", json.loads(md["raw_json"])
        except Exception:
            return "text", str(md["raw_json"])

    # 1) Keyed lookups (works whether docstore is a dict of composites or not)
    def _ds_get(key, default=None):
        if isinstance(docstore, dict):
            return docstore.get(key, default)
        return default

    doc_id = md.get("doc_id") or md.get("document_id") or md.get("source_id")
    if doc_id is not None:
        # JSON record
        rec_id = md.get("record_id") or md.get("entry_id") or md.get("key")
        if rec_id is not None:
            val = _ds_get(("json", doc_id, rec_id)) or _ds_get(f"{doc_id}:{rec_id}")
            if val is not None:
                if isinstance(val, (dict, list)):
                    return "json", val
                try:
                    return "json", json.loads(val)
                except Exception:
                    return "text", str(val)

        # PDF section
        section_id = md.get("section_id") or md.get("sec_id")
        if section_id is not None:
            val = _ds_get(("pdf_section", doc_id, section_id)) or _ds_get(f"{doc_id}:{section_id}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

        # PDF page fallback
        page = md.get("page") or md.get("page_index")
        if page is not None:
            try:
                p = int(page)
            except Exception:
                p = page
            val = _ds_get(("pdf_page", doc_id, p)) or _ds_get(f"{doc_id}:p{p}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

    # 2) Final fallback: return the chunk we searched/reranked on
    return "text", str(hit.get("text", ""))



def render_section_block(text: str):
    """Pretty full-section text block (no term slicing)."""
    import html
    # Precompute the escaped HTML so the f-string has no backslashes in expressions
    safe = html.escape(text or "").replace("\n", "<br/>")

    st.markdown(
        f"""
<div style="border:1px solid #e6e6e6;border-radius:10px;padding:12px 14px;background:#fffbe6;">
  <div style="font-size:0.92rem; line-height:1.55;">
    {safe}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# --------- PDF search helpers (drop near highlight utilities) ---------
def _is_pdf_token(term: str) -> bool:
    """Keep only tokens worth searching on-page."""
    t = (term or "").strip()
    if not t:
        return False
    # keep numbers with % or comparators (e.g. >99%)
    if re.search(r"[><≈~≤≥]?\s*\d{1,3}(?:[.,]\d+)?\s*[%％]", t):
        return True
    # keep short biomedical-ish uppercase tokens (genes, MT-TL1, POLG, ND5)
    if re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{2,}", t):
        return True
    # multi-word phrases are fragile on PDFs → split later
    if " " in t:
        return True
    # specific, non-generic single word (capitalized / contains digit / hyphen)
    if t[0].isupper() or any(ch.isdigit() for ch in t) or "-" in t:
        return True
    return False


def _explode_pdf_terms(terms: list[str]) -> list[str]:
    """
    Build robust search terms for PyMuPDF:
    - keep % numbers and gene-like tokens
    - split multi-word phrases
    - add hyphen/space variants for gene-like tokens
    - normalize punctuation variants
    """
    out: list[str] = []
    for t in terms:
        if not _is_pdf_token(t):
            continue

        base = unicodedata.normalize("NFKC", t).strip()
        if not base:
            continue

        # always keep the raw term
        out.append(base)

        # phrase → also search its content words (helps across line breaks)
        if " " in base:
            out.extend([p for p in re.split(r"\s+", base) if len(p) >= 3])

        # add hyphen / no-hyphen / space variants for gene-like tokens
        if "-" in base:
            out.append(base.replace("-", " "))
            out.append(base.replace("-", ""))

        # percent with/without space
        if re.search(r"\d+(?:[.,]\d+)?\s*[％%]", base):
            out.append(re.sub(r"\s*[％%]", "%", base))
            out.append(re.sub(r"\s*[％%]", " %", base))

    # longest-first & de-duplicate
    return list(dict.fromkeys(sorted(out, key=len, reverse=True)))




@st.cache_data(show_spinner=False)
def render_pdf_section_highlight_b64(pdf_path: str, page_index: int, para_text: str, zoom: float = 2.0) -> str | None:
    """One translucent rectangle over the section on the PDF page."""
    try:
        import fitz, io, base64
        from PIL import Image, ImageDraw
    except Exception:
        return None
    if not (pdf_path and os.path.exists(pdf_path) and para_text):
        return None

    try:
        with fitz.open(pdf_path) as doc:
            pidx = max(0, min(int(page_index) - 1 if page_index else 0, len(doc)-1))
            pg = doc[pidx]

            needle = para_text.strip().replace("\n", " ")[:700]
            rects = pg.search_for(
                needle,
                quads=False,
                flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES
            )
            if not rects:
                # fallback: first sentence
                import re
                s1 = re.split(r'(?<=[.!?])\s', needle, maxsplit=1)[0]
                if len(s1) > 20:
                    rects = pg.search_for(s1, quads=False, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES)

            m = fitz.Matrix(zoom, zoom)
            pix = pg.get_pixmap(matrix=m, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")

            if rects:
                X0 = min(r.x0 for r in rects); Y0 = min(r.y0 for r in rects)
                X1 = max(r.x1 for r in rects); Y1 = max(r.y1 for r in rects)
                x0, y0, x1, y1 = int(X0*zoom), int(Y0*zoom), int(X1*zoom), int(Y1*zoom)
                from PIL import ImageDraw
                overlay = Image.new("RGBA", img.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay, "RGBA")
                draw.rectangle((x0-4, y0-4, x1+4, y1+4), fill=(255,255,0,60), outline=(220,180,0,200), width=3)
                img = Image.alpha_composite(img, overlay)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            import base64
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def show_pdf_section_highlight(md: dict, section_text: str, answer_en: str):
    """
    On the PDF page:
      • highlight up to 2 best-matching sentences with a translucent block
      • also draw small boxes for important terms inside those sentences
    """
    pdf = md.get("source_file")
    if not (pdf and os.path.exists(pdf)):
        return

    # 1) pick best sentence(s) from the full section, driven by answer_en anchors
    top_sents = pick_best_sentences(section_text or "", answer_en or "", top_k=2)

    # 2) build term list from the answer (same as HTML side)
    #    we pass nlp=None here; if you want spaCy NER terms too, pass the loaded nlp
    terms = extract_terms(query_en="", answer_en=answer_en or "", nlp=None)

    # 3) render combined overlay (sentences + term boxes)
    b64 = render_pdf_sentence_and_term_overlay_b64(
        pdf_path=pdf,
        page_index=md.get("page") or 1,
        sentences=top_sents,
        terms=terms,
        zoom=2.0,
    )
    if b64:
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />',
            unsafe_allow_html=True,
        )



def render_sources_panel(
    hits: List[Dict[str, Any]],
    docstore: Any,
    query_en: str,
    answer_en: str,
    expanded: bool = False,
    nlp=None,
    embedder: Optional[SentenceTransformer] = None,
    show_raw: bool = False,
):
    if not hits:
        return

    # sort by score descending (prefer adjusted score, then reranker, then raw)
    def _score(h):
        return float(h.get("_adj_score", h.get("rerank_score", h.get("score", 0.0))))

    hits_sorted = sorted(hits, key=_score, reverse=True)

    # Compute terms once for all sources
    terms = extract_terms(query_en, answer_en, nlp=nlp)

    with st.expander(f"Sources ({len(hits_sorted)})", expanded=expanded):
        for i, h in enumerate(hits_sorted, 1):
            md = h.get("metadata", {}) or {}
            title = pretty_citation(md) or md.get("title") or md.get("source") or f"Source {i}"
            url = md.get("url") or md.get("source_url")
            score = h.get("_adj_score", h.get("rerank_score", h.get("score", 0.0)))

            with st.expander(f"[{i}] {title} — score {score:.3f}", expanded=False):
                if url:
                    st.caption(url)

                kind, content = resolve_full_chunk(h, docstore)
                pdf_path = md.get("source_file")

                # 1) PDF FIRST (sentence+term overlay if possible)
                if pdf_path and str(pdf_path).lower().endswith(".pdf") and os.path.exists(pdf_path):
                    # Use the same “best sentences” selection as the HTML panel
                    picked = []
                    if kind == "text" and content and embedder is not None:
                        try:
                            picked = select_supporting_sentences(
                                content, answer_en=answer_en or "", terms=terms or [],
                                nlp=nlp, embedder=embedder, top_k=2, min_overlap_terms=1
                            )
                        except Exception:
                            picked = []

                    # Draw sentence blocks + term boxes
                    b64 = render_pdf_sentence_and_term_overlay_b64(
                        pdf_path=pdf_path,
                        page_index=md.get("page") or 1,
                        sentences=picked or pick_best_sentences(content or "", answer_en or "", top_k=2),
                        terms=terms,
                        zoom=2.0,
                    )
                    if b64:
                        st.markdown(
                            f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />',
                            unsafe_allow_html=True,
                        )

                # 2) PRETTY HTML (no raw chunk)
                if kind == "text" and content:
                    # Mark only the picked sentences as blocks, and mark terms inside
                    if embedder is not None:
                        try:
                            picked = select_supporting_sentences(
                                content, answer_en=answer_en or "", terms=terms or [],
                                nlp=nlp, embedder=embedder, top_k=2, min_overlap_terms=1
                            )
                        except Exception:
                            picked = []
                    html_block = render_text_with_sentence_blocks(content, picked or [], terms or [])
                    if html_block:
                        st.markdown(html_block, unsafe_allow_html=True)
                    else:
                        render_section_block(content)

                elif kind == "json":
                    js = content if isinstance(content, (dict, list)) else {}
                    pretty = json.dumps(js, ensure_ascii=False, indent=2)
                    ex = excerpt_around(pretty, terms, window=800)
                    st.markdown(highlight_html(ex, terms), unsafe_allow_html=True)
                else:
                    raw = str(content or h.get("text", "") or "")
                    ex = excerpt_around(raw, terms, window=800)
                    st.markdown(highlight_html(ex, terms), unsafe_allow_html=True)

                # --- DEBUG: raw chunk dump ---
                if show_raw:
                    with st.expander("Raw chunk (debug)", expanded=False):
                        try:
                            if kind == "json":
                                st.code(json.dumps(content, ensure_ascii=False, indent=2), language="json")
                            else:
                                raw_txt = str(content or h.get("text", "") or "")
                                # Use a text_area so long chunks are easier to browse
                                st.text_area("raw", raw_txt, height=220)
                        except Exception:
                            st.text("Unavailable.")

                with st.expander("Record metadata", expanded=False):
                    st.json(md)





# ----------------------------
# Context assembly
# ----------------------------
def trim_to_token_budget(text: str, max_tokens: Optional[int]) -> str:
    if not max_tokens:
        return text
    if not _enc:
        return text[: max_tokens * 4]  # rough fallback: ~4 chars/token
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])


def build_messages(query: str, hits: List[Dict[str, Any]], system_prompt: str, num_ctx: int) -> List[Dict[str, str]]:
    blocks = []
    for i, h in enumerate(hits, 1):
        src = pretty_citation(h["metadata"])
        blocks.append(f"[{i}] {src}\n{h['text']}")
    ctx = "\n\n".join(blocks)
    # keep ~60% of context window for prompt+context to leave space for the answer
    ctx_budget = int(num_ctx * 0.6) if num_ctx else None
    ctx = trim_to_token_budget(ctx, ctx_budget)

    user_prompt = (
        "Use only the context to answer the question. If the answer is not present, say you don’t know.\n"
        "Cite sources with bracketed numbers [1], [2] matching the context blocks.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_messages_llm_only(query: str, system_prompt: str, chat_history: List[Dict[str, str]], history_turns: int = 3):
    conv_pairs = []
    for m in chat_history[-history_turns*2:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = trim_to_token_budget("\n".join(conv_pairs), max_tokens=256)

    user_prompt = (
        "Answer the user's question from your general knowledge. "
        "Do NOT use any external context.\n"
        "Do NOT include citations, bracketed numbers like [1], [2], or a References section.\n"
        "If the user greets or asks conversationally, respond politely.\n\n"
        f"Conversation (recent):\n{conversation}\n\n"
        f"Question: {query}\nAnswer:"
    )
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_messages_hybrid(
    query: str,
    hits: List[Dict[str, Any]],
    system_prompt: str,
    num_ctx: int,
    chat_history: List[Dict[str, str]],
    use_ids: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    # keep only selected ids if provided
    use = hits
    if use_ids:
        use = [h for i, h in enumerate(hits, 1) if i in use_ids]

    blocks = []
    for i, h in enumerate(use, 1):
        src = pretty_citation(h["metadata"])
        blocks.append(f"[{i}] {src}\n{h['text']}")
    ctx = "\n\n".join(blocks)
    ctx = trim_to_token_budget(ctx, int(num_ctx * 0.5) if num_ctx else None)

    conv_pairs = []
    for m in chat_history[-6:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = trim_to_token_budget("\n".join(conv_pairs), max_tokens=256)

    user_prompt = (
        "Use the documentary context IF it is relevant; otherwise answer from your general knowledge. "
        "Cite sources with bracketed numbers [1], [2] ONLY if you used the context. "
        "Never fabricate citations.\n\n"
        f"Conversation (recent):\n{conversation}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages



def build_messages_with_history(
    query: str,
    hits: List[Dict[str, Any]],
    system_prompt: str,
    num_ctx: int,
    chat_history: List[Dict[str, str]],
    history_turns: int = 3,
) -> List[Dict[str, str]]:
    conv_pairs = []
    for m in chat_history[-history_turns*2:]:
        if m["role"] in ("user", "assistant"):
            role = "Utilisateur" if m["role"] == "user" else "Assistant"
            conv_pairs.append(f"{role}: {m['content']}")
    conversation = "\n".join(conv_pairs)
    conversation = trim_to_token_budget(conversation, max_tokens=256)

    blocks = []
    for i, h in enumerate(hits, 1):
        src = pretty_citation(h["metadata"])
        blocks.append(f"[{i}] {src}\n{h['text']}")
    ctx = "\n\n".join(blocks)
    ctx_budget = int(num_ctx * 0.55) if num_ctx else None
    ctx = trim_to_token_budget(ctx, ctx_budget)

    user_prompt = (
        "Use only the documentary context to answer. "
        "If the information is not present, say that you do not know.\n"
        "Cite the sources with numbers in square brackets [1], [2], etc.\n\n"
        f"Conversation context (recent):\n{conversation}\n\n"
        f"Documentary context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_concat_query(
    chat_history: List[Dict[str, str]],
    user_input: str,
    max_prev_user_turns: int = 2,
    include_last_assistant: bool = True,
    token_budget: int = 256,
) -> str:
    """
    Deterministically builds a retrieval query by concatenating the last
    user turns (and optionally the last assistant reply) with the current input.
    This avoids relying on an LLM to rewrite the question.
    """
    parts: List[str] = []

    # Optionally include the last assistant turn (often contains the key entity)
    if include_last_assistant:
        for m in reversed(chat_history):
            if m.get("role") == "assistant":
                # Take only a short snippet to keep it lean
                a = (m.get("content") or "").strip()
                if a:
                    parts.append(a[:400])
                break

    # Include up to N previous user questions (most recent first)
    prev_users = [m.get("content", "") for m in chat_history if m.get("role") == "user"]
    for t in reversed(prev_users[-max_prev_user_turns:]):
        t = (t or "").strip()
        if t:
            parts.append(t)

    # Finally the current user input
    parts.append(user_input.strip())

    # Join and trim to a small token budget for the embedder
    combined = " \n".join(parts).strip()
    combined = trim_to_token_budget(combined, max_tokens=token_budget)
    return combined


# ----------------------------
# LLM providers
# ----------------------------
# @st.cache_resource(show_spinner=False)
# def get_llama_cpp(gen_cfg: Dict[str, Any]) -> Optional["Llama"]:
#     if not _LLAMACPP_OK:
#         return None
#     model_path = gen_cfg.get("model_path")
#     if not model_path or not os.path.exists(model_path):
#         return None
#     num_ctx = int(gen_cfg.get("num_ctx", 4096))
#     n_gpu_layers = int(gen_cfg.get("n_gpu_layers", 0))  # CPU by default
#     n_threads = int(gen_cfg.get("n_threads", 8))
#     n_batch = int(gen_cfg.get("n_batch", 256))
#     llm = Llama(
#         model_path=model_path,
#         n_ctx=num_ctx,
#         n_gpu_layers=n_gpu_layers,
#         n_threads=n_threads,
#         n_batch=n_batch,
#         logits_all=False,
#         verbose=False,
#     )
#     return llm
#
#
# def generate_with_llama_cpp(messages: List[Dict[str, str]], gen_cfg: Dict[str, Any]) -> Optional[str]:
#     llm = get_llama_cpp(gen_cfg)
#     if llm is None:
#         return None
#     temperature = float(gen_cfg.get("temperature", 0.2))
#     max_tokens = int(gen_cfg.get("max_tokens", 400))
#     try:
#         out = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
#         return out["choices"][0]["message"]["content"]
#     except Exception:
#         return None
@st.cache_resource(show_spinner=False)
def get_llama_cpp_model(
    model_path: str,
    num_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    n_batch: int,
    verbose: bool = False,
):
    if not _LLAMACPP_OK or not model_path or not os.path.exists(model_path):
        return None
    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_ctx=num_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        logits_all=False,
        verbose=verbose,
    )

def generate_with_llama_cpp(messages, gen_cfg):
    llm = get_llama_cpp_model(
        model_path=gen_cfg.get("model_path"),
        num_ctx=int(gen_cfg.get("num_ctx", 4096)),
        n_gpu_layers=int(gen_cfg.get("n_gpu_layers", 0)),
        n_threads=int(gen_cfg.get("n_threads", 8)),
        n_batch=int(gen_cfg.get("n_batch", 256)),
        verbose=False,
    )
    if llm is None:
        return None
    out = llm.create_chat_completion(
        messages=messages,
        temperature=float(gen_cfg.get("temperature", 0.2)),
        max_tokens=int(gen_cfg.get("max_tokens", 400)),
    )
    return out["choices"][0]["message"]["content"]


@st.cache_resource(show_spinner=False)
def get_hf_pipe(gen_cfg: Dict[str, Any]):
    if not _HF_OK:
        return None
    model_id = gen_cfg.get("model") or "microsoft/Phi-3.5-mini-instruct"
    quant = str(gen_cfg.get("quantization", "4bit")).lower()
    load_4bit = (quant == "4bit")
    load_8bit = (quant == "8bit")
    dtype = torch.float16
    device_map = gen_cfg.get("device_map", "auto")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=bool(gen_cfg.get("trust_remote_code", False)))
    bnb_cfg = None
    if load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=bnb_cfg,
        load_in_8bit=load_8bit if not load_4bit else False,
        trust_remote_code=bool(gen_cfg.get("trust_remote_code", False)),
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        torch_dtype=dtype,
        device_map=device_map,
    )


def messages_to_text_prompt(messages: List[Dict[str, str]]) -> str:
    out = []
    for m in messages:
        role = m["role"].capitalize()
        out.append(f"{role}: {m['content']}")
    out.append("Assistant:")
    return "\n\n".join(out)


def generate_with_hf(messages: List[Dict[str, str]], gen_cfg: Dict[str, Any]) -> Optional[str]:
    pipe = get_hf_pipe(gen_cfg)
    if pipe is None:
        return None
    temperature = float(gen_cfg.get("temperature", 0.2))
    max_tokens = int(gen_cfg.get("max_tokens", 400))
    prompt = messages_to_text_prompt(messages)
    try:
        res = pipe(
            prompt,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )
        text = res[0]["generated_text"]
        return text.split("Assistant:", 1)[-1].strip()
    except Exception:
        return None


def rewrite_query_with_history(
    chat_history: List[Dict[str, str]],
    user_input: str,
    provider: str,
    gen_cfg: Dict[str, Any],
    max_turns: int = 4,
) -> str:
    """
    Returns a standalone French query that resolves pronouns/coreferences.
    Falls back to the original user_input on error.
    """
    recent = chat_history[-max_turns*2:] if chat_history else []

    sys_prompt = (
        "Tu réécris la question suivante pour qu’elle soit autonome en conservant la langue d’origine (français). "
        "Résous les références comme « il/elle/ce/cela/ceci/ça » en remplaçant par l’entité correcte d’après la conversation. "
        "N’ajoute aucune information qui n’est pas déjà présente. "
        "Si la question est déjà autonome, renvoie-la telle quelle. "
        "Réponds uniquement par la question réécrite, sans explication ni guillemets."
    )

    messages = [{"role": "system", "content": sys_prompt}]
    for m in recent:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    try:
        if provider == "llama_cpp":
            llm = get_llama_cpp_model(
                model_path=gen_cfg.get("model_path"),
                num_ctx=int(gen_cfg.get("num_ctx", 2048)),
                n_gpu_layers=int(gen_cfg.get("n_gpu_layers", 0)),
                n_threads=int(gen_cfg.get("n_threads", 8)),
                n_batch=int(gen_cfg.get("n_batch", 256)),
                verbose=False,
            )
            if llm is None:
                return user_input
            out = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=96)
            text = out["choices"][0]["message"]["content"]
        else:
            tmp_cfg = {
                "model": gen_cfg.get("model"),
                "quantization": gen_cfg.get("quantization", "4bit"),
                "device_map": gen_cfg.get("device_map", "auto"),
                "temperature": 0.0,
                "max_tokens": 96,
                "trust_remote_code": gen_cfg.get("trust_remote_code", False),
            }
            text = generate_with_hf(messages, tmp_cfg) or user_input

        text = (text or "").strip()
        # Strip quotes/brackets if the model wraps the output
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("«") and text.endswith("»")):
            text = text[1:-1].strip()
        return text or user_input
    except Exception:
        return user_input



# ----------------------------
# Source preview (PDF/TXT/JSON)
# ----------------------------
def _pdf_iframe_html(pdf_path: str, height: int = 600) -> str:
    with open(pdf_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'


# def render_source_preview(md: Dict[str, Any]):
#     sf = md.get("source_file")
#     src_type = md.get("source")
#     if not sf or not os.path.exists(sf):
#         st.info("No local source file available.")
#         st.json(md)
#         return
#
#     p = Path(sf)
#     st.write(f"Local file: {p.name}")
#     st.caption(str(p))
#
#     # PDFs
#     if p.suffix.lower() == ".pdf":
#         try:
#             html = _pdf_iframe_html(str(p))
#             st.components.v1.html(html, height=650, scrolling=True)
#         except Exception:
#             st.info("Could not render PDF inline. You can open it locally.")
#     # Plain text
#     elif p.suffix.lower() in {".txt", ".md", ".csv", ".tsv"}:
#         try:
#             with open(p, "r", encoding="utf-8", errors="ignore") as f:
#                 txt = f.read()
#             st.text_area("Text preview", txt[:50_000], height=350)
#         except Exception:
#             st.info("Could not render text preview.")
#     # JSON-like
#     elif p.suffix.lower() == ".json":
#         try:
#             with open(p, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             st.json(data)
#         except Exception:
#             st.info("Could not render JSON preview.")
#     else:
#         st.info("Preview not supported for this file type.")
#     # Always show metadata
#     with st.expander("Record metadata", expanded=False):
#         st.json(md)
def render_source_preview(md: Dict[str, Any], key_suffix: str):
    sf = md.get("source_file")
    if not sf or not os.path.exists(sf):
        st.info("No local source file available.")
        with st.expander("Record metadata", expanded=False):
            st.json(md)
        return

    p = Path(sf)
    st.write(f"Local file: {p.name}")
    st.caption(str(p))

    # Download only for PDFs (no inline embed)
    if p.suffix.lower() == ".pdf":
        try:
            size = os.path.getsize(p)
            st.caption(f"PDF size: {size/1024/1024:.2f} MB")
        except Exception:
            pass
        with open(p, "rb") as fh:
            st.download_button(
                "Download PDF",
                data=fh,
                file_name=p.name,
                mime="application/pdf",
                key=f"dl_{key_suffix}",
            )

    # Text files: tiny preview + optional full
    elif p.suffix.lower() in {".txt", ".md", ".csv", ".tsv"}:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            st.text(txt[:5000])
            if len(txt) > 5000 and st.checkbox("Show full file", key=f"fulltxt_{key_suffix}"):
                st.text_area("Full text", txt[:100_000], height=300)
        except Exception:
            st.info("Could not render text preview.")

    # JSON: opt-in to show
    elif p.suffix.lower() == ".json":
        try:
            if st.checkbox("Show JSON", value=False, key=f"showjson_{key_suffix}"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.json(data)
        except Exception:
            st.info("Could not render JSON preview.")

    else:
        st.info("Preview not supported for this file type.")

    with st.expander("Record metadata", expanded=False):
        st.json(md)


# -------- Images helper ---------------
def st_safe_image(img, caption=None, use_container_width=True):
    """
    Accept bytes / BytesIO / PIL image / numpy array / valid file path or URL.
    Silently skips on invalid inputs to avoid MediaFileStorageError.
    """
    try:
        # bytes or BytesIO
        if isinstance(img, (bytes, bytearray, io.BytesIO)):
            if isinstance(img, io.BytesIO):
                img = img.getvalue()
            st.image(img, caption=caption, use_container_width=use_container_width)
            return
        # PIL Image
        if isinstance(img, Image.Image):
            st.image(img, caption=caption, use_container_width=use_container_width)
            return
        # string path/URL
        if isinstance(img, str):
            # Only try to display if it's an existing local file or a URL
            if (img.startswith("http://") or img.startswith("https://")) or os.path.exists(img):
                st.image(img, caption=caption, use_container_width=use_container_width)
            else:
                st.caption("Image not available.")
            return
        # numpy array
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                st.image(img, caption=caption, use_container_width=use_container_width)
                return
        except Exception:
            pass
        # Fallback: do nothing
        st.caption("Image not available.")
    except MediaFileStorageError:
        st.caption("Image not available.")
    except Exception:
        st.caption("Image not available.")

# -------- Highlighting utilities --------
import html
import re
try:
    import fitz  # PyMuPDF for PDF highlight previews
    _PDF_OK = True
except Exception:
    _PDF_OK = False

BIO_TOKEN_RE = re.compile(
    r"(?:m\.\d+[ACGT]>\d*[ACGT])|"
    r"(?:c\.\d+[ACGT]>\d*[ACGT])|"
    r"(?:p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2})|"
    r"(?:MT-[A-Z0-9]+)|"
    r"(?:rs\d+)|"
    r"(?:MELAS|LHON|MERRF|PKAN|POLG|PANK2|ND[1-6])",
    re.I
)

GENE_SYM_RE = re.compile(r"\b[A-Z0-9]{2,10}(?:-[A-Z0-9]{1,5})?\b")

PERCENT_RE = re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s*%\b")         # 99%, 1.5%
NUMBER_UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|µg|ug|g|mL|L|mm|cm|kg|years?|yrs?)\b", re.I)
BARE_NUMBER_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
QUOTE_RE = re.compile(r"[\"“”‘’«»]([^\"“”‘’«»]{3,80})[\"“”‘’«»]")  # phrases inside quotes

# --- Numeric patterns: allow optional comparators and unicode % ---
_COMPARATORS = (">", "≥", "<", "≤", "~", "≈")
_SPACE_VARIANTS = ["", " ", "\u00A0", "\u202F", "\u2009"]  # none, space, NBSP, narrow NBSP, thin space
_PERCENT_CHARS = ["%", "％"]  # ASCII percent + full-width percent

PERCENT_RE = re.compile(
    r"[><≈~≤≥]?\s*\d{1,3}(?:[.,]\d+)?\s*[%％]",  # 0–100% (or 100+ if you like)
    re.I
)

NUMBER_UNIT_RE = re.compile(
    r"[><≈~≤≥]?\s*\d+(?:[.,]\d+)?\s*(?:mg|µg|ug|g|mL|L|mm|cm|kg|years?|yrs?|yo|y|months?|days?)\b",
    re.I
)

def expand_numeric_variants(s: str) -> list[str]:
    """For '99%' -> ['99%', '>99%', '≥99%', '<99%', '≤99%', '~99%', '≈99%'] and also the bare variant."""
    s = s.strip()
    out = [s]
    bare = s
    if bare and bare[0] in _COMPARATORS:
        bare = bare.lstrip("".join(_COMPARATORS)).lstrip()
        out.append(bare)
    for c in _COMPARATORS:
        if not s.startswith(c):
            out.append(f"{c}{bare}")
    return list(dict.fromkeys(out))



def _nfkd(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "")

# very small domain-generic stop nouns (kept tiny on purpose)
_UNINFORMATIVE_NOUNS = {
    "patients","individual","individuals","subjects","people","person","persons",
    "cases","case","children","adults","age","year","years","study","studies",
    "treatment","management","therapy","guideline","report","reports","evidence",
    "disease","syndrome","condition","mutation","variant","gene","genes","protein"
}

def _is_generic_single_word(t: str) -> bool:
    """Drop lowercase single tokens that look generic."""
    s = (t or "").strip()
    if not s or (" " in s):             # only single words
        return False
    if any(ch.isdigit() for ch in s):   # keep token with digits (e.g., G6PD)
        return False
    if "-" in s:                         # keep hyphenated terms
        return False
    if s[0].isupper():                   # keep proper-name-ish tokens
        return False
    return s.lower() in _UNINFORMATIVE_NOUNS

def is_informative_chunk(nc) -> bool:
    """Keep multi-word/contentful chunks; drop generic single nouns."""
    text = nc.text.strip().strip(".,;:()[]{}")
    if len(text) < 3:
        return False
    tokens = [t for t in nc if not t.is_space]
    is_multi = (" " in text)
    # Single-word chunk must look "specific": capitalized/digit/hyphen/proper noun
    if not is_multi:
        t0 = tokens[0]
        if (not t0.text[0].isupper()) and (not any(ch.isdigit() for ch in t0.text)) and ("-" not in t0.text) and (t0.pos_ != "PROPN"):
            return False
        if t0.lemma_.lower() in _UNINFORMATIVE_NOUNS:
            return False
    # Multi-word: drop if all lemmas are generic
    lowers = {t.lemma_.lower() for t in tokens if t.is_alpha}
    if is_multi and lowers and lowers.issubset(_UNINFORMATIVE_NOUNS):
        return False
    return True


# --- Robust normalization for both TEXT and TERMS ---
_SOFT = "\u00AD"            # soft hyphen
_ZW   = "\u200B\u200C\u200D\u2060"  # zero-width / word joiners
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # hyphen, NB hyphen, figure/en/em/minus

def _normalize_text_for_match(s: str) -> str:
    """Fix PDF artefacts so literals can match consistently."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # remove soft/zero-width
    s = s.translate({ord(c): None for c in _SOFT + _ZW})
    # unify unicode dashes to ASCII hyphen
    s = s.translate({ord(c): ord("-") for c in _DASHES})
    # join hyphenated line-breaks: MT-\nTL1 -> MT-TL1
    s = re.sub(r"-\s*\n\s*", "-", s)
    # collapse remaining newlines to a single space
    s = re.sub(r"\s*\n\s*", " ", s)
    # collapse all whitespace runs
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_term(t: str) -> str:
    """Same normalization for terms so they align with normalized text."""
    return _normalize_text_for_match(t)

def _term_pattern(t: str) -> str:
    """
    Escape a term for regex but make hyphens tolerant to variants and spaces.
    After normalization, we still allow dash + optional spaces just in case.
    """
    if not t:
        return ""
    esc = re.escape(t)
    esc = esc.replace(r"\-", r"[-\u2011\u2013\u2014]\s*")  # tolerant hyphen
    return esc



def extract_terms(query_en: str, answer_en: str, extra_terms: list[str] | None = None, nlp=None) -> list[str]:
    terms = set()
    q = (answer_en or "")
    q2 = (query_en or "")

    # 0) Strict biomedical tokens (your existing patterns)
    for s in (q, q2):
        for m in BIO_TOKEN_RE.findall(s): terms.add(m)
        for g in GENE_SYM_RE.findall(s):
            if len(g) >= 3 and not g.isdigit():
                terms.add(g)

    # 1) Quoted phrases (keep raw and NFKD)
    for s in (q, q2):
        for m in QUOTE_RE.findall(s):
            t = m.strip()
            if len(t) >= 3:
                terms.add(t)
                terms.add(_nfkd(t))

    # 2) Percent / number+unit (with comparator expansions)
    for s in (q, q2):
        for m in PERCENT_RE.findall(s):
            for v in expand_numeric_variants(m):
                terms.add(v); terms.add(_nfkd(v))
        for m in NUMBER_UNIT_RE.findall(s):
            for v in expand_numeric_variants(m):
                terms.add(v); terms.add(_nfkd(v))

    # 3) NER entities + noun chunks (filtered)
    if nlp is not None:
        for s in (q, q2):
            if not s: continue
            try:
                doc = nlp(s)
                for ent in doc.ents:
                    t = ent.text.strip().strip(".,;:()[]{}")
                    if len(t) >= 3:
                        terms.add(t); terms.add(_nfkd(t))
                for nc in getattr(doc, "noun_chunks", []):
                    if is_informative_chunk(nc):
                        t = nc.text.strip().strip(".,;:()[]{}")
                        terms.add(t); terms.add(_nfkd(t))
            except Exception:
                pass

    # 4) user extras (optional)
    if extra_terms:
        for t in extra_terms:
            if t:
                terms.add(t); terms.add(_nfkd(t))

    # 5) Phrase-over-single collapse: keep multi-word phrases; drop singles they contain.
    phrases = [t for t in terms if " " in t]
    singles = [t for t in terms if " " not in t]
    singles_keep = set(singles)
    for p in phrases:
        # whole-word containment: drop 'eye' when 'eye of the tiger' exists
        wb = re.compile(rf"(?i)(?<!\w){re.escape(p)}(?!\w)")
        for s in list(singles_keep):
            # if single appears as a whole word inside the phrase, drop it
            if re.search(rf"(?i)(?<!\w){re.escape(s)}(?!\w)", p):
                singles_keep.discard(s)
    final_terms = set(phrases) | singles_keep

    # Final prune: remove generic single words universally
    final_terms = {t for t in final_terms if not _is_generic_single_word(t)}

    # Sort longest-first so alternation prefers full phrases
    return sorted(final_terms, key=len, reverse=True)


def build_pdf_terms_from_answer(query_en: str, answer_en: str, nlp=None) -> list[str]:
    """
    Use the same term extractor as the HTML highlighter, then add
    PDF-friendly variants (e.g., '80%' and '80 %', 'MT-TL1' and 'MT TL1').
    """
    base = extract_terms(query_en, answer_en, nlp=nlp)

    variants: list[str] = []
    for t in base:
        v = [t]

        # percent with/without space
        if re.search(r"\d+(?:[.,]\d+)?\s*[％%]", t):
            v.append(re.sub(r"\s*[％%]", "%", t))
            v.append(re.sub(r"\s*[％%]", " %", t))

        # hyphen variants for gene-like tokens (MT-TL1 -> "MT TL1" too)
        if "-" in t:
            v.append(t.replace("-", " "))
            v.append(t.replace("-", ""))  # rarely helps, but cheap

        # normalize unicode forms (NFKC) as separate probes
        v.append(unicodedata.normalize("NFKC", t))

        # dedupe while preserving order
        seen = set()
        vv = []
        for x in v:
            if x and x not in seen:
                vv.append(x); seen.add(x)
        variants.extend(vv)

    # final dedupe, longest first so search_for hits the most specific first
    variants = list(dict.fromkeys(sorted(variants, key=len, reverse=True)))
    return variants



def excerpt_around(text: str, needles: list[str], window: int = 400) -> str:
    """Return a substring around the first occurrence of any needle (normalized)."""
    work = _normalize_text_for_match(text or "")
    if not work:
        return ""
    # normalize needles too
    norm_needles = [_normalize_term(n) for n in (needles or []) if n]
    lw = work.lower()
    for n in norm_needles:
        i = lw.find(n.lower())
        if i != -1:
            start = max(0, i - window//2)
            end = min(len(work), i + len(n) + window//2)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(work) else ""
            return prefix + work[start:end] + suffix
    return (work[:window] + "…") if len(work) > window else work



def highlight_html(text: str, terms: list[str]) -> str:
    """HTML with <mark> highlights; robust to PDF artefacts and hyphenation."""
    if not text:
        return ""
    # Normalize the working copy of the TEXT
    work = _normalize_text_for_match(text)
    # Normalize TERMS the same way, then build tolerant patterns
    norm_terms = []
    for t in terms or []:
        nt = _normalize_term(t)
        if nt and len(nt) >= 2:
            norm_terms.append(_term_pattern(nt))
    # Nothing to highlight
    if not norm_terms:
        return html.escape(work).replace("\n", "<br/>")

    safe = html.escape(work)
    rx = re.compile(r"(" + "|".join(norm_terms) + r")", re.I)
    marked = rx.sub(r"<mark>\1</mark>", safe)
    return marked.replace("\n", "<br/>")


# ---------- Sentence selection (spaCy + embeddings) and rendering ----------

def _split_sentences_spacy(text: str, nlp) -> list[str]:
    """Prefer spaCy for sentence splitting; fallback to a light regex."""
    t = text or ""
    if not t:
        return []
    if nlp is not None:
        try:
            doc = nlp(t)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                return sents
        except Exception:
            pass
    # fallback
    return re.split(r'(?<=[.!?])\s+(?=[A-Z(0-9])', t)

def _embed_matrix(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype="float32")
    return a @ b.T  # already normalized

def select_supporting_sentences(
    chunk_text: str,
    answer_en: str,
    terms: list[str],
    nlp,
    embedder: SentenceTransformer,
    top_k: int = 2,
    min_overlap_terms: int = 1,
) -> list[str]:
    """Pick the top_k sentences by cosine similarity to the answer, with a soft boost for term overlap."""
    if not chunk_text:
        return []
    sents = _split_sentences_spacy(chunk_text, nlp)
    if not sents:
        return []
    try:
        ans_vec = _embed_matrix([answer_en or ""], embedder)
        sent_vec = _embed_matrix(sents, embedder)
        sims = _cos(sent_vec, ans_vec).ravel()
    except Exception:
        sims = np.zeros(len(sents), dtype="float32")

    # term-overlap soft boost
    term_rx = None
    if terms:
        pats = [re.escape(_normalize_term(t)) for t in terms if t and len(t) >= 2]
        if pats:
            term_rx = re.compile("(?i)" + "|".join(pats))

    def overlap_count(s):
        if not term_rx:
            return 0
        ns = _normalize_text_for_match(s)
        return len(set(m.group(0).lower() for m in term_rx.finditer(ns)))

    overlaps = np.array([overlap_count(s) for s in sents], dtype="float32")
    sims = sims + 0.05 * overlaps  # tiny boost

    idxs = np.argsort(-sims)
    if min_overlap_terms > 0 and overlaps.max() >= min_overlap_terms:
        idxs = [i for i in idxs if overlaps[i] >= min_overlap_terms][:top_k]
    else:
        idxs = list(idxs[:top_k])

    return [sents[i] for i in idxs]

def render_text_with_sentence_blocks(full_text: str, picked_sents: list[str], terms: list[str]) -> str:
    """
    Wrap picked sentences in a soft block (<div class="sent-block">) and <mark> any terms within.
    """
    if not full_text:
        return ""
    norm_full = _normalize_text_for_match(full_text)
    spans = []

    for s in picked_sents:
        ns = _normalize_text_for_match(s)
        if not ns:
            continue
        i = norm_full.lower().find(ns.lower())
        if i == -1:
            continue
        # tolerant pattern to grab the original slice
        tokens = [re.escape(tok) for tok in ns.split()]
        pat = r"(?i)" + r"\s+".join(tokens)
        m = re.search(pat, full_text)
        if not m:
            m = re.search(re.escape(s), full_text)
        if m:
            spans.append((m.start(), m.end()))

    # merge overlaps
    spans.sort()
    merged = []
    for a, b in spans:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    parts = []
    last = 0
    for a, b in merged:
        pre = full_text[last:a]
        mid = full_text[a:b]
        parts.append(html.escape(pre).replace("\n", "<br/>"))
        parts.append(f'<div class="sent-block">{highlight_html(mid, terms)}</div>')
        last = b
    parts.append(html.escape(full_text[last:]).replace("\n", "<br/>"))
    return "".join(parts)

def render_pdf_sentence_highlights(md: dict, picked_sents: list[str], zoom: float = 2.0):
    """
    Draw translucent rectangles over the selected sentences on the PDF page and render as an <img>.
    """
    try:
        import fitz, io, base64
        from PIL import Image, ImageDraw
    except Exception:
        return
    pdf_path = md.get("source_file")
    if not (pdf_path and os.path.exists(pdf_path)):
        return
    page = md.get("page") or 1
    try:
        pidx = int(page) - 1
    except Exception:
        pidx = 0

    with fitz.open(pdf_path) as doc:
        pidx = max(0, min(pidx, len(doc)-1))
        pg = doc[pidx]
        rects = []
        for s in picked_sents:
            s_norm = _normalize_text_for_match(s)[:800]
            if not s_norm:
                continue
            found = pg.search_for(
                s_norm,
                flags=getattr(fitz, "TEXT_DEHYPHENATE", 0) | getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0),
                quads=False,
            )
            if not found and len(s_norm) > 60:
                half = s_norm[: len(s_norm)//2]
                found = pg.search_for(half, flags=getattr(fitz, "TEXT_DEHYPHENATE", 0) | getattr(fitz, "TEXT_PRESERVE_LIGATURES", 0), quads=False)
            rects.extend(found)

        m = fitz.Matrix(zoom, zoom)
        pix = pg.get_pixmap(matrix=m, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        for r in rects:
            x0, y0, x1, y1 = int(r.x0*zoom), int(r.y0*zoom), int(r.x1*zoom), int(r.y1*zoom)
            draw.rectangle((x0-2, y0-2, x1+2, y1+2), fill=(255, 255, 0, 55), outline=(220, 180, 0, 200), width=3)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />', unsafe_allow_html=True)



# --- Canonicalization & sentence matching helpers ---

try:
    from rapidfuzz.fuzz import token_set_ratio as _tfuzz  # optional
    _FUZZ_OK = True
except Exception:
    _FUZZ_OK = False

_WS = re.compile(r"\s+")
# Hyphen line-breaks you often get in PDFs: "MT-\nTL1" → "MTTL1" (but we'll allow flexible match)
_HLB = re.compile(r"-\s*\n\s*")

def _strip_accents(s: str) -> str:
    import unicodedata as _ud
    return "".join(ch for ch in _ud.normalize("NFKD", s or "") if not _ud.combining(ch))

def canon(s: str) -> str:
    """Lowercase, strip accents, collapse spaces, remove hyphen line-breaks."""
    if not s:
        return ""
    s = _HLB.sub("", s)
    s = _strip_accents(s)
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s

def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter that survives PDFs reasonably well."""
    # First, join hyphen line-breaks (already in canon), then split on punctuation.
    # Keep short/empty sentences out.
    raw = text or ""
    # Split while keeping reasonable chunks
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", raw)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            out.append(p)
    if not out:
        out = [raw]
    return out

def _flex_gene_pattern(sym: str) -> str:
    """
    Turn 'MT-TL1' -> 'M\s*[-–]?\s*T\s*[-–]?\s*TL1' (but simpler: allow optional hyphen/space between tokens).
    For general gene-like tokens we allow optional hyphen/space breaks.
    """
    # split on hyphens and join with '[-–]?\s*'
    parts = re.split(r"[-–\s]+", sym)
    if len(parts) <= 1:
        # single token, just escape
        return re.escape(sym)
    return r"\s*[-–]?\s*".join(map(re.escape, parts))

def _flex_percent_pattern(p: str) -> str:
    """Allow optional comparators and unicode/narrow spaces around percent sign."""
    # Extract the numeric core (e.g. '80%','>80 %','≈80％')
    m = re.search(r"\d{1,3}(?:[.,]\d+)?", p)
    if not m:
        return re.escape(p)
    num = m.group(0)
    # Optional comparator at start; optional spacing; percent chars; optional spacing
    return rf"(?:[><≈~≤≥]\s*)?{re.escape(num)}\s*[％%]"

def _flex_phrase_pattern(q: str) -> str:
    """For multi-word phrases: allow flexible whitespace and optional hyphen joins."""
    words = [w for w in re.split(r"\s+", q.strip()) if w]
    pats = []
    for w in words:
        # allow unit words (keep strict), for gene-like allow optional hyphen
        if re.match(r"[A-Za-z]+-\w+", w):
            pats.append(_flex_gene_pattern(w))
        else:
            pats.append(re.escape(w))
    return r"\s+".join(pats)

def extract_anchor_patterns(answer_en: str) -> list[re.Pattern]:
    """
    Build a small set of robust regex patterns that must anchor the answer:
    - HGVS-like variants (from your BIO_TOKEN_RE)
    - Gene-like tokens (GENE_SYM_RE)
    - Percents (PERCENT_RE)
    - Quoted phrases (QUOTE_RE)
    """
    anchors: list[str] = []
    s = answer_en or ""

    # BIO tokens and genes (dedupe)
    anchors += list({m for m in BIO_TOKEN_RE.findall(s) if m})
    anchors += list({g for g in GENE_SYM_RE.findall(s) if g})

    # percents (e.g. 80%, >80 %)
    anchors += list({m for m in PERCENT_RE.findall(s) if m})

    # quoted phrases
    anchors += list({m.strip() for m in QUOTE_RE.findall(s) if len(m.strip()) >= 3})

    pats: list[re.Pattern] = []
    for a in anchors:
        a_stripped = a.strip()
        if not a_stripped:
            continue
        if "%" in a_stripped or "％" in a_stripped:
            rx = _flex_percent_pattern(a_stripped)
        elif "-" in a_stripped:
            rx = _flex_gene_pattern(a_stripped)
        elif " " in a_stripped:
            rx = _flex_phrase_pattern(a_stripped)
        else:
            rx = re.escape(a_stripped)
        try:
            pats.append(re.compile(rx, re.I))
        except Exception:
            # fallback to escaped
            pats.append(re.compile(re.escape(a_stripped), re.I))
    return pats

def score_sentence(sent: str, pats: list[re.Pattern], answer_en: str) -> float:
    """Weighted score = anchor hits + optional fuzzy similarity."""
    if not sent:
        return 0.0
    s = sent
    base = 0.0
    for p in pats:
        if p.search(s):
            # Give a bit more weight to % and HGVS-like substrings
            rx = p.pattern
            w = 2.0 if ("%" in rx or "％" in rx or "m\\." in rx or "c\\." in rx) else 1.0
            base += w
    # Optional fuzzy boost
    if _FUZZ_OK and answer_en:
        try:
            base += 0.02 * _tfuzz(canon(sent), canon(answer_en))  # small weight
        except Exception:
            pass
    # Length penalty (very short or very long sentences get a tiny nudge)
    L = len(sent)
    if L < 40: base *= 0.9
    if L > 600: base *= 0.9
    return base

def pick_best_sentences(section_text: str, answer_en: str, top_k: int = 3) -> list[str]:
    if not section_text:
        return []
    pats = extract_anchor_patterns(answer_en or "")
    sents = split_sentences(section_text)
    scored = [(score_sentence(s, pats, answer_en), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [s for (sc, s) in scored[:top_k] if sc > 0]
    # If nothing scored > 0, take the first sentence to avoid empty highlight
    return picked or sents[:1]

def render_text_with_sentence_highlights(full_text: str, selected_sents: list[str]):
    """Render full section with only selected sentences wrapped in <mark>."""
    if not full_text:
        return
    safe = html.escape(full_text)
    # escape selected sentences and mark them (longest first)
    pats = [re.escape(s.strip()) for s in sorted(selected_sents, key=len, reverse=True) if s.strip()]
    if pats:
        rx = re.compile("(" + "|".join(pats) + ")", re.I)
        highlighted = rx.sub(r"<mark>\1</mark>", safe)
    else:
        highlighted = safe
    html_block = highlighted.replace("\n", "<br/>")
    st.markdown(
        '<div style="border:1px solid #e6e6e6;border-radius:10px;padding:12px 14px;background:#fffff6;">{}</div>'.format(html_block),
        unsafe_allow_html=True,
    )



# def show_pdf_section_highlight(md: dict, section_text: str, answer_en: str):
#     """
#     Highlight up to top-2 informative sentences on the PDF page.
#     Falls back gracefully if search misses.
#     """
#     pdf = md.get("source_file")
#     if not (pdf and os.path.exists(pdf)):
#         return
#
#     top_sents = pick_best_sentences(section_text, answer_en, top_k=2)
#
#     b64 = render_pdf_section_highlight_b64(
#         pdf_path=pdf,
#         page_index=md.get("page") or 1,
#         para_text=" ".join(top_sents)[:900] if top_sents else (section_text or "")[:900],
#         zoom=2.0,
#     )
#     if b64:
#         st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" />', unsafe_allow_html=True)




def render_pdf_highlight(md: dict, terms: list[str], key_suffix: str, max_boxes: int = 12):
    """
    PDF page preview with outline-only highlights, embedded as a data-URI image.
    More robust token-level search to handle ligatures, hyphenation, and line breaks.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageDraw
        import io, base64
    except Exception:
        st.info("PyMuPDF/PIL not installed; cannot render PDF highlights.")
        return

    sf = md.get("source_file")
    if not sf or not os.path.exists(sf):
        st.info("No local PDF available.")
        return

    page = md.get("page")
    if page is None:
        st.info("No page metadata to render a PDF preview.")
        return

    # Page index (accepts one-based from metadata)
    try:
        pidx = int(page)
        if pidx > 0 and md.get("_one_based_pages", True):
            pidx -= 1
    except Exception:
        pidx = 0

    # Build robust token list for PDF search
    pdf_terms = _explode_pdf_terms(terms)
    if not pdf_terms:
        st.caption("No highlightable terms detected.")
        return

    # Search flags: preserve ligatures, de-hyphenate, ignore case
    # (TEXT_IGNORECASE exists in recent PyMuPDF; if not, we fall back without it)
    flags = 0
    for name in ("TEXT_PRESERVE_LIGATURES", "TEXT_DEHYPHENATE", "TEXT_PRESERVE_WHITESPACE", "TEXT_IGNORECASE"):
        flags |= getattr(fitz, name, 0)

    try:
        with fitz.open(sf) as doc:
            pidx = max(0, min(pidx, len(doc) - 1))
            pg = doc[pidx]

            rects = []
            for t in pdf_terms:
                if len(rects) >= max_boxes:
                    break
                # short guard to avoid flooding with super-common tokens
                if len(t) < 3:
                    continue
                for r in pg.search_for(t, quads=False, flags=flags):
                    rects.append(r)
                    if len(rects) >= max_boxes:
                        break

            # Render to PNG
            zoom = 2.0
            m = fitz.Matrix(zoom, zoom)
            pix = pg.get_pixmap(matrix=m, alpha=False)
            png_bytes = pix.tobytes("png")

        # Draw OUTLINE ONLY around hits
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")

        def to_px(r):
            return (int(r.x0 * zoom), int(r.y0 * zoom), int(r.x1 * zoom), int(r.y1 * zoom))

        for r in rects:
            x0, y0, x1, y1 = to_px(r)
            pad = 2
            box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            draw.rectangle(box, outline=(255, 200, 0, 255), width=3)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" alt="PDF page preview" style="width:100%;height:auto;" />',
            unsafe_allow_html=True,
        )
        st.caption(f"PDF page preview (page {pidx + 1})")
    except Exception as e:
        st.info(f"PDF preview error: {e}")


@st.cache_data(show_spinner=False)
def render_pdf_sentence_and_term_overlay_b64(
    pdf_path: str,
    page_index: int,
    sentences: list[str],
    terms: list[str] | None = None,
    zoom: float = 2.0,
) -> str | None:
    """
    Renders a PDF page with:
      • one translucent rectangle per selected sentence (spanning all its lines)
      • thin outline boxes for important 'terms' (optional)
    Returns a base64-encoded PNG for <img src="data:..."> embedding.
    """
    try:
        import fitz
        from PIL import Image, ImageDraw
        import io, base64, unicodedata, re
    except Exception:
        return None

    if not (pdf_path and os.path.exists(pdf_path)):
        return None

    # Normalize helper (to survive hyphenation/ligatures/odd spaces)
    def _nfkc(s: str) -> str:
        s = unicodedata.normalize("NFKC", s or "")
        s = re.sub(r"-\s*\n\s*", "-", s)        # join hyphen line-breaks
        s = re.sub(r"\s*\n\s*", " ", s)         # collapse newlines
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # PDF search flags: be tolerant to hyphenation/ligatures/whitespace/case
    flags = 0
    for name in ("TEXT_PRESERVE_LIGATURES", "TEXT_DEHYPHENATE", "TEXT_PRESERVE_WHITESPACE", "TEXT_IGNORECASE"):
        flags |= getattr(fitz, name, 0)

    try:
        with fitz.open(pdf_path) as doc:
            pidx = max(0, min(int(page_index) - 1 if page_index else 0, len(doc)-1))
            pg = doc[pidx]

            # Collect rectangles for sentences (each may span multiple lines)
            sent_rect_groups: list[list[fitz.Rect]] = []
            for s in (sentences or []):
                s_norm = _nfkc(s)
                if not s_norm or len(s_norm) < 10:
                    continue

                # Try full sentence; if that fails, try a robust prefix chunk
                rects = pg.search_for(s_norm, quads=False, flags=flags)
                if not rects:
                    # use first ~80–120 chars as a robust key
                    # (keeps us within a single paragraph line group)
                    chunk = s_norm[:120]
                    # don't end in middle of a token (roughly)
                    chunk = re.sub(r"\W+\w*$", "", chunk)
                    if len(chunk) > 20:
                        rects = pg.search_for(chunk, quads=False, flags=flags)

                if rects:
                    sent_rect_groups.append(rects)

            # Collect rectangles for term tokens (optional)
            term_rects: list[fitz.Rect] = []
            if terms:
                # explode into PDF-friendly tokens (your helper)
                pdf_terms = _explode_pdf_terms(terms)
                for t in pdf_terms:
                    if len(t) < 2:
                        continue
                    # limit over-highlighting by skipping ultra-common words
                    if t.lower() in {"the", "of", "and"}:
                        continue
                    found = pg.search_for(t, quads=False, flags=flags)
                    term_rects.extend(found)

            # Render page → RGBA Image
            m = fitz.Matrix(zoom, zoom)
            pix = pg.get_pixmap(matrix=m, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")

            # --- draw on a separate transparent overlay, then composite ---
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay, "RGBA")

            def to_px(r: fitz.Rect) -> tuple[int, int, int, int]:
                return (int(r.x0 * zoom), int(r.y0 * zoom), int(r.x1 * zoom), int(r.y1 * zoom))

            # 1) Sentence highlight: per line, light fill, tiny vertical pad
            for rects in sent_rect_groups:
                for r in rects:
                    x0, y0, x1, y1 = to_px(r)
                    pad_x, pad_y = 2, 1  # very small so we don’t bleed into other lines
                    box = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
                    # very light “marker” fill + thin outline
                    odraw.rectangle(box, fill=(255, 255, 0, 28), outline=(220, 180, 0, 180), width=2)

            # 2) Term boxes (optional)
            for r in term_rects[:60]:
                x0, y0, x1, y1 = to_px(r)
                pad = 2
                odraw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad),
                                outline=(255, 200, 0, 220), width=2)

            # composite once at the end (preserves the underlying text)
            img = Image.alpha_composite(img, overlay)

            # Encode as base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None




# ----------------------------
# Sanity check helpers
# ----------------------------
CITE_RE = re.compile(r"\s*\[\d+\]")  # matches ' [1]', ' [23]' anywhere
REFS_RE = re.compile(r"(?is)\n?\s*references\s*:.*$")  # strip a tailing 'References:' section

def strip_citations(text: str) -> str:
    if not text:
        return text
    # Remove any [number] citations
    text = CITE_RE.sub("", text)
    # Remove trailing 'References:' block if the model added one
    text = REFS_RE.sub("", text)
    return text.strip()


# ----------------------------
# App
# ----------------------------
def main():
    st.set_page_config(page_title="Local RAG Q&A", layout="wide")

    st.markdown("""
    <style>
    .sent-block{
      background:#fffbe6;
      border:1px solid #f0e6b3;
      border-radius:8px;
      padding:6px 8px;
      display:inline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hardcoded relative path
    logo_path = "../assets/logo.png"

    col1, col2 = st.columns([1, 8], vertical_alignment="center")
    with col1:
        st_safe_image(logo_path, use_container_width=True)
    with col2:
        st.title("Mito Chat")

    cfg = read_config()
    nlp = load_ner((cfg.get("ner", {}) or {}).get("model", "en_core_sci_sm"))

    index_dir = cfg["paths"]["index_dir"]
    retr = cfg.get("retrieval", {})
    gen = cfg.get("generation", {}) or {}

    # Load FAISS index and embedder
    with st.spinner("Loading index..."):
        index, docstore, meta = load_index(index_dir)
    embedder = load_embedder(meta.get("embed_model_name") or cfg["embedding"]["model_name"])

    # Try subindices
    gr_ix, gr_ds = load_subindex(index_dir, "genereviews")
    cl_ix, cl_ds = load_subindex(index_dir, "clinvar")
    mt_ix, mt_ds = load_subindex(index_dir, "mitocarta")
    sub = {"genereviews": (gr_ix, gr_ds), "clinvar": (cl_ix, cl_ds), "mitocarta": (mt_ix, mt_ds)}

    have_sub = any(ix is not None and ds is not None for ix, ds in sub.values())

    # Load ID maps
    id_maps = load_id_maps(index_dir)

    # Retrieval config
    per_source_k = retr.get("per_source_k", {"genereviews": 40, "clinvar": 6, "mitocarta": 6})
    final_k = int(retr.get("final_k", retr.get("rerank_top_k", 8)))
    multi_stage = bool(retr.get("multi_stage", False))

    # Translation config
    trans_cfg = cfg.get("translation", {}) or {}
    translator = get_translator(
        device=trans_cfg.get("device"),
        max_new_tokens=int(trans_cfg.get("max_new_tokens", 512)),
        cache_bust="v1",  # <-- change this string when you change translate.py
    )

    # Load Spacy
    nlp = load_ner(cfg.get("ner", {}).get("model", "en_core_sci_sm"))

    # Sidebar settings
    with st.sidebar:
        # Small inline action just above the input bar
        if st.button("Nouveau chat", key="btn_new_chat_inline", help="Nouvelle conversation",
                     use_container_width=True):
            st.session_state.chat = []
            st.rerun()

        st.caption(f"Compute device: {pick_device()}")

        # Query rewritting
        st.divider()
        use_rewrite = st.checkbox("Use query rewriting", value=True)
        show_debug = st.checkbox("Show rewrite/debug info", value=False)
        st.session_state["use_rewrite"] = use_rewrite
        st.session_state["show_debug"] = show_debug

        st.header("Retrieval")
        top_k = st.slider("Vector top_k", 5, 100, int(retr.get("top_k", 20)))
        enable_reranker = st.checkbox("Enable reranker", value=bool(retr.get("enable_reranker", True)))
        rerank_top_k = st.slider("Rerank to", 1, 20, int(retr.get("rerank_top_k", 5)))
        reranker_model = st.text_input("Reranker model", retr.get("reranker_model", "BAAI/bge-reranker-base"))

        st.header("Generation")
        provider = st.selectbox("Provider", ["llama_cpp", "transformers"], index=0)

        # Common generation knobs
        temperature = st.slider("Temperature", 0.0, 1.0, float(gen.get("temperature", 0.2)), 0.05)
        max_tokens = st.slider("Max answer tokens", 64, 2048, int(gen.get("max_tokens", 400)), 32)
        num_ctx = st.slider("Context window (tokens)", 1024, 8192, int(gen.get("num_ctx", 4096)), 256)
        system_prompt = st.text_area("System prompt", gen.get("system_prompt", ""), height=80)


        st.divider()
        st.caption("llama.cpp options (GGUF, in-process)")
        model_path = st.text_input("GGUF model_path", gen.get("model_path", "models/qwen2.5-3b-instruct-q4_k_m.gguf"))
        n_threads = st.number_input("CPU threads (llama.cpp)", min_value=1, max_value=64, value=int(gen.get("n_threads", 8)))
        n_batch = st.number_input("Batch size (llama.cpp)", min_value=16, max_value=512, value=int(gen.get("n_batch", 256)))
        n_gpu_layers = st.number_input("n_gpu_layers (0=CPU)", min_value=0, max_value=128, value=int(gen.get("n_gpu_layers", 0)))

        st.caption("Transformers options (optional)")
        hf_model = st.text_input("HF model id or path", gen.get("model", "microsoft/Phi-3.5-mini-instruct"))
        quant = st.selectbox("Quantization (Transformers)", ["4bit", "8bit", "none"], index=0)
        device_map = st.text_input("device_map", gen.get("device_map", "auto"))

        from llama_cpp import llama_print_system_info

        st.divider()
        st.caption("llama.cpp diagnostics")
        st.code(llama_print_system_info(), language="text")
        if provider == "llama_cpp":
            st.caption(f"Using llama.cpp with n_gpu_layers={int(n_gpu_layers)} and model={model_path}")
        else:
            st.caption("Provider is transformers — n_gpu_layers is ignored here.")

    # --- Chat state ---
    if "chat" not in st.session_state:
        # Each item: {"role": "user"|"assistant", "content": str, "hits": Optional[List[hit_dict]]}
        st.session_state.chat = []


    # --- Chat UI ---
    # st.subheader("Chat")

    live = st.session_state.get("_rendering_live", False)

    # Display existing messages (including sources for assistant turns)
    # for i, m in enumerate(st.session_state.chat):
    #     with st.chat_message(m["role"]):
    #         st.markdown(m["content"])
    #         if m["role"] == "user" and st.session_state.get("show_debug", True) and m.get("rewrite"):
    #             with st.expander("Requête utilisée pour la recherche", expanded=False):
    #                 st.code(m["rewrite"], language="text")
    #
    #         # NEW: pretty sources panel for assistant turns (after rerun)
    #         if m["role"] == "assistant" and m.get("hits"):
    #             q_en_hist = m.get("rewrite") or m.get("rewrite_input_en") or ""
    #             ans_en_hist = m.get("answer_en") or ""
    #             render_sources_panel(
    #                 m["hits"], docstore,
    #                 query_en=q_en_hist, answer_en=ans_en_hist,
    #                 expanded=False, nlp=nlp, embedder=embedder
    #             )
    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "user" and st.session_state.get("show_debug", True) and m.get("rewrite"):
                with st.expander("Requête utilisée pour la recherche", expanded=False):
                    st.code(m["rewrite"], language="text")
                with st.expander("Router decision", expanded=False):
                    st.json(m["router"])

            # Pretty sources panel for assistant turns (AFTER reruns)
            if m["role"] == "assistant" and m.get("hits"):
                # If we're also rendering a live assistant message in this same run,
                # skip sources for the very last assistant (avoid duplicate/flicker).
                if live and i == len(st.session_state.chat) - 1:
                    continue

                q_en_hist = m.get("rewrite") or m.get("rewrite_input_en") or ""
                ans_en_hist = m.get("answer_en") or ""
                render_sources_panel(
                    m["hits"], docstore,
                    query_en=q_en_hist, answer_en=ans_en_hist,
                    expanded=False, nlp=nlp, embedder=embedder,
                    show_raw=st.session_state.get("show_debug", False),
                )

    # Accept new user input
    user_input = st.chat_input("Posez une question sur les maladies mitochondriales…")
    if user_input:
        # 1) Add user message and render immediately (original FR)
        st.session_state.chat.append({"role": "user", "content": user_input})
        user_idx = len(st.session_state.chat) - 1
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) Translate FR -> EN for rewriting/retrieval/generation
        try:
            q_en = translate_text(user_input, "fr", "en", translator)
        except Exception:
            q_en = user_input  # safe fallback
        st.session_state.chat[user_idx]["rewrite_input_en"] = q_en

        # 3) Rewrite in English (you already loaded English prompts)
        standalone_q, rw_debug = rewrite_query_llm_from_yaml(
            st.session_state.get("chat", []),
            q_en,  # English input to the rewriter
            provider,
            {
                "model_path": model_path,
                "num_ctx": int(num_ctx),
                "n_gpu_layers": int(n_gpu_layers),
                "n_threads": int(n_threads),
                "n_batch": int(n_batch),
                "model": hf_model,
                "quantization": quant,
                "device_map": device_map,
                "trust_remote_code": gen.get("trust_remote_code", False),
            },
            prompts=rewrite_prompts,  # English prompts
        )

        st.session_state.chat[user_idx]["rewrite"] = standalone_q
        st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug

        # 4) Retrieval uses the English rewritten query
        with st.spinner("Collecte d'information…"):
            struct_hits = structured_lookup_first(standalone_q, id_maps, docstore, limit_per_key=3)
            if multi_stage and have_sub:
                dense_hits = search_multi_stage(
                    standalone_q,
                    embedder=embedder,
                    per_source_indices=sub,
                    per_source_k=per_source_k,
                    final_k=rerank_top_k,  # use your slider
                    enable_reranker=enable_reranker,
                    reranker_model=reranker_model,
                    retr_cfg={**retr, "final_k": rerank_top_k, "rerank_top_k": rerank_top_k},
                )
            else:
                dense_hits = search(
                    standalone_q, index, docstore, embedder,
                    top_k=top_k,
                    enable_reranker=enable_reranker,
                    rerank_top_k=rerank_top_k,
                    reranker_model=reranker_model,
                    retr_cfg={**retr, "rerank_top_k": rerank_top_k},
                    id_maps=id_maps,
                )
            # Merge + dedupe
            hits, seen = [], set()
            for h in struct_hits + dense_hits:
                k = _hit_key(h)
                if k in seen:
                    continue
                hits.append(h)
                seen.add(k)

        # 5) Decide mode with router (post-retrieval light agent)
        route = call_router_llm(
            router_prompt,
            query=standalone_q,
            hits=hits,
            provider=provider,
            gen_cfg={
                "model_path": model_path,
                "num_ctx": int(num_ctx),
                "n_gpu_layers": int(n_gpu_layers),
                "n_threads": int(n_threads),
                "n_batch": int(n_batch),
                "model": hf_model,
                "quantization": quant,
                "device_map": device_map,
                "trust_remote_code": gen.get("trust_remote_code", False),
            },
        )


        mode = route.get("mode", "RAG" if hits else "LLM")
        use_ids = route.get("use_ids", [])

        # (optional) guardrails that force RAG on identifier queries
        has_identifier = bool(detect_identifiers(standalone_q)) or bool(detect_identifiers(user_input))
        has_struct = 'struct_hits' in locals() and bool(struct_hits)
        if has_identifier or has_struct:
            mode = "RAG"

        # Persist for this user turn (so it survives st.rerun)
        st.session_state.chat[user_idx]["router"] = {
            "raw": route,
            "mode": mode,
            "use_ids": use_ids,
            "hits_count": len(hits),
        }

        if st.session_state.get("show_debug", True):
            with st.expander("Router decision", expanded=False):
                st.json(st.session_state.chat[user_idx]["router"])


        # 6) Build messages based on mode
        if mode == "LLM":
            messages = build_messages_llm_only(standalone_q, system_prompt, st.session_state.chat)
        elif mode == "HYBRID":
            messages = build_messages_hybrid(standalone_q, hits, system_prompt, num_ctx, st.session_state.chat,
                                             use_ids=use_ids)
        else:  # RAG
            # Use your strict doc-only builder, but optionally filter to use_ids
            if use_ids:
                selected = [h for i, h in enumerate(hits, 1) if i in use_ids]
                if selected:
                    messages = build_messages_with_history(standalone_q, selected, system_prompt, num_ctx,
                                                           st.session_state.chat)
                else:
                    messages = build_messages_with_history(standalone_q, hits, system_prompt, num_ctx,
                                                           st.session_state.chat)
            else:
                messages = build_messages_with_history(standalone_q, hits, system_prompt, num_ctx,
                                                       st.session_state.chat)

        # # 6) Generate in EN, translate to FR, display FR, and attach EN rewrite for debug
        # with st.chat_message("assistant"):
        #     if not hits:
        #         msg_en = "I couldn't find relevant passages. Try rephrasing."
        #         try:
        #             msg_fr = translate_text(msg_en, "en", "fr", translator)
        #         except Exception:
        #             msg_fr = "Je ne trouve pas de passages pertinents. Essayez de reformuler."
        #         st.markdown(msg_fr)
        #         st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite shown later in expander
        #         st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
        #         st.session_state.chat.append({"role": "assistant", "content": msg_fr, "hits": []})
        #         # st.rerun()
        #
        #     if provider == "llama_cpp":
        #         llm = get_llama_cpp_model(
        #             model_path=model_path,
        #             num_ctx=int(num_ctx),
        #             n_gpu_layers=int(n_gpu_layers),
        #             n_threads=int(n_threads),
        #             n_batch=int(n_batch),
        #             verbose=False,
        #         )
        #         # Buffer English; show French only at the end
        #         full_text_en = ""
        #         try:
        #             for token in llm.create_chat_completion(messages=messages, temperature=float(temperature),
        #                                                     max_tokens=int(max_tokens), stream=True):
        #                 delta = token["choices"][0]["delta"].get("content", "")
        #                 if delta:
        #                     full_text_en += delta
        #         except Exception:
        #             out = llm.create_chat_completion(messages=messages, temperature=float(temperature),
        #                                              max_tokens=int(max_tokens))
        #             full_text_en = out["choices"][0]["message"]["content"]
        #
        #         # Decide if citations are allowed
        #         allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
        #
        #         if not allow_citations:
        #             full_text_en = strip_citations(full_text_en)
        #
        #         # Translate EN -> FR for display
        #         try:
        #             full_text_fr = translate_text(full_text_en or "", "en", "fr", translator)
        #         except Exception:
        #             full_text_fr = full_text_en or "Échec de traduction/génération."
        #
        #         # Only attach sources if we actually used RAG/HYBRID
        #         used_hits = [] if mode == "LLM" else hits
        #
        #         st.markdown(full_text_fr)
        #
        #         # NEW: pretty sources with highlights (pass EN query/answer)
        #         if used_hits:
        #             render_sources_panel(used_hits, docstore, query_en=standalone_q, answer_en=full_text_en, nlp=nlp, embedder=embedder)
        #
        #         st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite
        #         st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
        #         st.session_state.chat.append({
        #             "role": "assistant",
        #             "content": full_text_fr,
        #             "hits": used_hits,
        #             "answer_en": full_text_en,  # <- store EN answer for highlighting
        #             "rewrite": standalone_q,  # <- store EN rewritten query
        #             "mode": mode,
        #         })
        #
        #
        #     else:
        #         answer_en = generate_with_hf(messages, {
        #             "temperature": float(temperature),
        #             "max_tokens": int(max_tokens),
        #             "model": hf_model,
        #             "quantization": quant,
        #             "device_map": device_map,
        #             "trust_remote_code": gen.get("trust_remote_code", False),
        #         })
        #
        #         allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
        #         if not allow_citations:
        #             answer_en = strip_citations(answer_en)
        #
        #         try:
        #             answer_fr = translate_text(answer_en or "", "en", "fr", translator)
        #         except Exception:
        #             answer_fr = answer_en or "Échec de génération."
        #         st.markdown(answer_fr)
        #
        #         # Only attach sources if we actually used RAG/HYBRID
        #         used_hits = [] if mode == "LLM" else hits
        #
        #         # NEW: pretty sources with highlights
        #         if used_hits:
        #             render_sources_panel(used_hits, docstore, query_en=standalone_q, answer_en=answer_en or "", nlp=nlp, embedder=embedder)
        #
        #         st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite
        #         st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
        #         st.session_state.chat.append({
        #             "role": "assistant",
        #             "content": answer_fr or "",
        #             "hits": used_hits,
        #             "answer_en": answer_en or "",  # <- store EN answer for highlighting
        #             "rewrite": standalone_q,  # <- store EN rewritten query
        #             "mode": mode,
        #         })
        #
        # # st.rerun()
        # --- 6) Generate in EN, translate to FR, display FR, and attach EN rewrite for debug
        # Pin a "live render" flag so the history loop won't also render sources for the last turn in this same run
        # st.session_state["_rendering_live"] = True

        # with st.chat_message("assistant"):
        #     # pin placeholders inside the bubble
        #     ans_area = st.empty()  # where we'll put the final French answer
        #     src_holder = st.container()  # where we'll render the Sources panel later
        #
        #     if not hits:
        #         msg_en = "I couldn't find relevant passages. Try rephrasing."
        #         try:
        #             msg_fr = translate_text(msg_en, "en", "fr", translator)
        #         except Exception:
        #             msg_fr = "Je ne trouve pas de passages pertinents. Essayez de reformuler."
        #         ans_area.markdown(msg_fr)
        #         # store turn (no sources)
        #         st.session_state.chat.append({"role": "assistant", "content": msg_fr, "hits": []})
        #         st.session_state["_rendering_live"] = False
        #     else:
        #         if provider == "llama_cpp":
        #             llm = get_llama_cpp_model(
        #                 model_path=model_path,
        #                 num_ctx=int(num_ctx),
        #                 n_gpu_layers=int(n_gpu_layers),
        #                 n_threads=int(n_threads),
        #                 n_batch=int(n_batch),
        #                 verbose=False,
        #             )
        #             full_text_en = ""
        #             try:
        #                 for token in llm.create_chat_completion(
        #                         messages=messages,
        #                         temperature=float(temperature),
        #                         max_tokens=int(max_tokens),
        #                         stream=True
        #                 ):
        #                     delta = token["choices"][0]["delta"].get("content", "")
        #                     if delta:
        #                         full_text_en += delta
        #             except Exception:
        #                 out = llm.create_chat_completion(
        #                     messages=messages,
        #                     temperature=float(temperature),
        #                     max_tokens=int(max_tokens)
        #                 )
        #                 full_text_en = out["choices"][0]["message"]["content"]
        #
        #             allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
        #             if not allow_citations:
        #                 full_text_en = strip_citations(full_text_en)
        #
        #             try:
        #                 full_text_fr = translate_text(full_text_en or "", "en", "fr", translator)
        #             except Exception:
        #                 full_text_fr = full_text_en or "Échec de traduction/génération."
        #
        #             # Show the answer now, *inside* this bubble
        #             ans_area.markdown(full_text_fr)
        #
        #             used_hits = [] if mode == "LLM" else hits
        #
        #             # Render Sources *in the pinned container* (so it won't appear near the input)
        #             if used_hits:
        #                 with src_holder:
        #                     render_sources_panel(
        #                         used_hits, docstore,
        #                         query_en=standalone_q,
        #                         answer_en=full_text_en,
        #                         expanded=False,
        #                         nlp=nlp,
        #                         embedder=embedder
        #                     )
        #
        #             # Persist the turn to history (so next run will render it in the history list)
        #             st.session_state.chat.append({
        #                 "role": "assistant",
        #                 "content": full_text_fr,
        #                 "hits": used_hits,
        #                 "answer_en": full_text_en,
        #                 "rewrite": standalone_q,
        #                 "mode": mode,
        #             })
        #             st.session_state["_rendering_live"] = False
        #
        #         else:
        #             answer_en = generate_with_hf(messages, {
        #                 "temperature": float(temperature),
        #                 "max_tokens": int(max_tokens),
        #                 "model": hf_model,
        #                 "quantization": quant,
        #                 "device_map": device_map,
        #                 "trust_remote_code": gen.get("trust_remote_code", False),
        #             }) or ""
        #
        #             allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
        #             if not allow_citations:
        #                 answer_en = strip_citations(answer_en)
        #
        #             try:
        #                 answer_fr = translate_text(answer_en, "en", "fr", translator)
        #             except Exception:
        #                 answer_fr = answer_en or "Échec de génération."
        #
        #             ans_area.markdown(answer_fr)
        #
        #             used_hits = [] if mode == "LLM" else hits
        #             if used_hits:
        #                 with src_holder:
        #                     render_sources_panel(
        #                         used_hits, docstore,
        #                         query_en=standalone_q,
        #                         answer_en=answer_en,
        #                         expanded=False,
        #                         nlp=nlp,
        #                         embedder=embedder
        #                     )
        #
        #             st.session_state.chat.append({
        #                 "role": "assistant",
        #                 "content": answer_fr,
        #                 "hits": used_hits,
        #                 "answer_en": answer_en,
        #                 "rewrite": standalone_q,
        #                 "mode": mode,
        #             })
        #             st.session_state["_rendering_live"] = False

        # --- Generate, commit to history, and rerun (no live rendering) ---
        with st.spinner("Préparation de la réponse…"):
            if not hits:
                msg_en = "I couldn't find relevant passages. Try rephrasing."
                try:
                    msg_fr = translate_text(msg_en, "en", "fr", translator)
                except Exception:
                    msg_fr = "Je ne trouve pas de passages pertinents. Essayez de reformuler."

                # Persist the turn (no sources), then rerun so it renders once in history
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": msg_fr,
                    "hits": [],
                })
                st.session_state.pop("_rendering_live", None)
                st.rerun()

            else:
                if provider == "llama_cpp":
                    llm = get_llama_cpp_model(
                        model_path=model_path,
                        num_ctx=int(num_ctx),
                        n_gpu_layers=int(n_gpu_layers),
                        n_threads=int(n_threads),
                        n_batch=int(n_batch),
                        verbose=False,
                    )

                    # Generate (non-stream to avoid any live output)
                    try:
                        out = llm.create_chat_completion(
                            messages=messages,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens)
                        )
                        full_text_en = out["choices"][0]["message"]["content"] or ""
                    except Exception:
                        full_text_en = ""

                    # Gate citations: only keep if we actually used docs
                    allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
                    if not allow_citations:
                        full_text_en = strip_citations(full_text_en)

                    # Translate EN → FR for display
                    try:
                        full_text_fr = translate_text(full_text_en, "en", "fr", translator)
                    except Exception:
                        full_text_fr = full_text_en or "Échec de traduction/génération."

                    used_hits = [] if mode == "LLM" else hits

                    # Persist to history; the history loop will render the message and Sources once
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": full_text_fr,
                        "hits": used_hits,
                        "answer_en": full_text_en,
                        "rewrite": standalone_q,
                        "mode": mode,
                        "router": st.session_state.chat[user_idx].get("router"),
                    })
                    st.session_state.pop("_rendering_live", None)
                    st.rerun()

                else:
                    # transformers branch
                    answer_en = generate_with_hf(messages, {
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "model": hf_model,
                        "quantization": quant,
                        "device_map": device_map,
                        "trust_remote_code": gen.get("trust_remote_code", False),
                    }) or ""

                    allow_citations = (mode in ("RAG", "HYBRID")) and bool(use_ids)
                    if not allow_citations:
                        answer_en = strip_citations(answer_en)

                    try:
                        answer_fr = translate_text(answer_en, "en", "fr", translator)
                    except Exception:
                        answer_fr = answer_en or "Échec de génération."

                    used_hits = [] if mode == "LLM" else hits

                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": answer_fr,
                        "hits": used_hits,
                        "answer_en": answer_en,
                        "rewrite": standalone_q,
                        "mode": mode,
                        "router": st.session_state.chat[user_idx].get("router"),
                    })
                    st.session_state.pop("_rendering_live", None)
                    st.rerun()


if __name__ == "__main__":
    main()
