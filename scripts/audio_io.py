"""
audio_io.py — tiny STT/TTS helpers for MitoChat.

Backends:
- STT: faster-whisper (default), Vosk (optional fully-offline fallback)
- TTS: Kokoro (default), pyttsx3 (CPU fallback)

Install (pick what you need):
    pip install faster-whisper onnxruntime  # STT (tiny/int8 recommended)
    pip install kokoro-tts onnxruntime      # TTS (Kokoro ONNX)
    pip install streamlit-mic-recorder      # mic capture widget
Optional:
    pip install vosk                        # offline STT fallback
    pip install pyttsx3                     # simple CPU TTS fallback

Kokoro models:
- Download an ONNX voice package (e.g., English multi-speaker).
  Example (directory layout):
    kokoro_models/
      kokoro-v1.onnx
      voices/
        af_bella.json
        am_michael.json
        ...
  Then set TTS.model_dir="kokoro_models" and TTS.voice="af_bella" in config.yaml

Author: you 🧑‍💻
"""

from __future__ import annotations
import io
import os, tempfile, wave, librosa
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# --------- Small device helper ----------
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


def _detect_format(b: bytes) -> str:
    if len(b) < 16: return "unknown"
    sig = b[:16]
    if sig[:4] == b"RIFF" and sig[8:12] == b"WAVE":
        return "wav"
    if sig[:3] == b"ID3" or (len(b) > 2 and sig[0] == 0xFF and (sig[1] & 0xE0) == 0xE0):
        return "mp3"
    if sig[:4] == b"OggS":
        return "ogg"
    if sig[:4] == b"\x1aE\xdf\xa3":  # Matroska/WebM
        return "webm"
    if sig[4:8] == b"ftyp":  # MP4/M4A
        return "m4a"
    return "unknown"


def _load_any_bytes_to_wav_path(b: bytes) -> tuple[str, float]:
    """
    Save incoming bytes (wav/mp3/ogg/webm/m4a) to a real WAV file on disk (16 kHz mono),
    returning (wav_path, duration_sec). Ensures the file handle is CLOSED before returning.
    """
    # 1) Decode to mono float32 at native sr (via either soundfile fast path or librosa+ffmpeg)
    fmt = _detect_format(b)
    if fmt == "wav":
        data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
    else:
        # Write bytes to a CLOSED temp path first (Windows-safe)
        fd_in, in_path = tempfile.mkstemp(suffix=f".{fmt if fmt != 'unknown' else 'bin'}")
        os.close(fd_in)
        try:
            with open(in_path, "wb") as f:
                f.write(b)
            y, sr = librosa.load(in_path, sr=None, mono=True)
            data = y.astype("float32")
        finally:
            try:
                os.remove(in_path)
            except Exception:
                pass

    # 2) Resample to 16 kHz mono
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    duration = float(len(data) / sr)

    # 3) Write PCM16 WAV to a CLOSED temp path (Windows-safe)
    fd_out, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_out)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wav16 = (data * 32767.0).astype("int16")
        wf.writeframes(wav16.tobytes())

    return wav_path, duration

# -----------------------
# Small utils
# -----------------------
def _wav_bytes_from_float32(wav: np.ndarray, sr: int = 24000) -> bytes:
    """
    Ensures float32 mono waveform -> PCM_16 WAV bytes.
    Cleans NaNs/Infs and normalizes quiet audio a bit.
    """
    import soundfile as sf

    x = np.asarray(wav, dtype="float32").squeeze()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        raise RuntimeError("TTS: empty waveform.")

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if 0.0 < peak < 0.2:
        x = x / (peak + 1e-9) * 0.95

    buf = io.BytesIO()
    sf.write(buf, x, samplerate=sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def _normalize_tts_backend(name: Optional[str]) -> str:
    n = (name or "").strip().lower()
    if n in ("kokoro_kpipeline", "kokoro_pipeline", "kpipeline"):
        return "kokoro_kpipeline"
    if n in ("pyttsx3",):
        return "pyttsx3"
    raise ValueError(f"Unknown TTS backend: {name}")

# ======================
# STT (speech -> text)
# ======================
@dataclass
class STTConfig:
    backend: str = "faster-whisper"  # "faster-whisper" | "vosk"
    model: str = "tiny.en"           # faster-whisper: "tiny", "base", "small", etc.
    compute_type: str = "int8"       # "int8" (very small), "int8_float16", "int8_dynamic", "float16", "float32"
    vad_filter: bool = True
    language: Optional[str] = None   # e.g., "fr", "en"; None = auto
    # vosk:
    vosk_model_dir: Optional[str] = None  # path to unpacked Vosk model


class SpeechToText:
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._init_backend()

    def _init_backend(self):
        if self.cfg.backend == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except Exception as e:
                raise RuntimeError(f"Install faster-whisper to use STT backend 'faster-whisper': {e}")
            device = pick_device()
            self._whisper = WhisperModel(self.cfg.model, device=device, compute_type=self.cfg.compute_type)
            self._vosk = None
        elif self.cfg.backend == "vosk":
            try:
                import vosk
            except Exception as e:
                raise RuntimeError(f"Install vosk to use STT backend 'vosk': {e}")
            if not self.cfg.vosk_model_dir or not os.path.isdir(self.cfg.vosk_model_dir):
                raise RuntimeError("Vosk backend requires a valid STT.vosk_model_dir.")
            import json
            self._vosk_model = vosk.Model(self.cfg.vosk_model_dir)
            self._vosk = True
            self._whisper = None
        else:
            raise ValueError(f"Unknown STT backend: {self.cfg.backend}")

    def transcribe(self, wav_bytes: bytes) -> Tuple[str, float]:
        # --- Build a Windows-safe temp WAV path first ---
        wav_path, duration = _load_any_bytes_to_wav_path(wav_bytes)

        try:
            if self._whisper is not None:
                segments, info = self._whisper.transcribe(
                    wav_path,
                    language=self.cfg.language,
                    vad_filter=self.cfg.vad_filter,
                    beam_size=1,
                    best_of=1,
                )
                text_parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
                return " ".join(p for p in text_parts if p).strip(), duration

            if self._vosk:
                import vosk, json
                import soundfile as sf
                data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                vosk.SetLogLevel(-1)
                rec = vosk.KaldiRecognizer(self._vosk_model, sr)
                wav16 = (data * 32767.0).astype("int16").tobytes()
                for i in range(0, len(wav16), 4000):
                    rec.AcceptWaveform(wav16[i:i + 4000])
                obj = json.loads(rec.FinalResult())
                return (obj.get("text") or "").strip(), duration

            return "", duration
        finally:
            # Clean up the temp file
            try:
                os.remove(wav_path)
            except Exception:
                pass


# ======================
# TTS (text -> speech)
# ======================
@dataclass
class TTSConfig:
    backend: str = "kokoro_kpipeline"  # "kokoro_onnx" | "kokoro_kpipeline" | "pyttsx3"
    model_dir: Optional[str] = None    # for ONNX (contains kokoro-v1.onnx + voices/) or hint for KPipeline
    voices_dir: Optional[str] = None   # only for KPipeline if your fork uses a separate voices dir
    lang_code: Optional[str] = None    # for KPipeline (e.g., "f", "a", …)
    voice: str = "ff_siwis"            # voice name
    speed: float = 1.0
    output_format: str = "wav"


class TextToSpeech:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._kpipeline = None     # PyTorch pipeline
        self._engine = None        # pyttsx3
        self._init_backend()

    # ----------- INIT -----------
    def _init_backend(self):
        backend = _normalize_tts_backend(self.cfg.backend)

        if backend == "kokoro_kpipeline":
            try:
                from kokoro import KPipeline  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Install the KPipeline package to use backend 'kokoro_kpipeline'.\n"
                    "Example: pip install kokoro soundfile torch\n"
                    f"Details: {e}"
                )

            # Some forks take environment hints for model/voices
            if self.cfg.model_dir:
                os.environ.setdefault("KOKORO_MODEL_DIR", self.cfg.model_dir)
            if self.cfg.voices_dir:
                os.environ.setdefault("KOKORO_VOICES_DIR", self.cfg.voices_dir)

            lang = self.cfg.lang_code or "f"  # default to French for your app
            self._kpipeline = KPipeline(lang_code=lang)
            self._kokoro = None
            self._engine = None
            return

        if backend == "pyttsx3":
            try:
                import pyttsx3  # type: ignore
            except Exception as e:
                raise RuntimeError("Install pyttsx3 to use backend 'pyttsx3':\n pip install pyttsx3\nDetails: " + str(e))
            eng = pyttsx3.init()
            # voice selection: substring match
            vname = (self.cfg.voice or "").strip()
            if vname:
                try:
                    for v in eng.getProperty("voices"):
                        nm = getattr(v, "name", "") or ""
                        id_ = getattr(v, "id", "") or ""
                        if vname.lower() in nm.lower() or vname.lower() in id_.lower():
                            eng.setProperty("voice", v.id)
                            break
                except Exception:
                    pass
            # rate mapping (words per minute)
            try:
                base = int(eng.getProperty("rate") or 200)
                eng.setProperty("rate", max(80, min(350, int(base * float(self.cfg.speed or 1.0)))))
            except Exception:
                pass
            self._engine = eng
            self._kokoro = None
            self._kpipeline = None
            return

        raise ValueError(f"Unknown TTS backend after normalization: {backend}")

    # ----------- SYNTHESIS -----------
    def synthesize(self, text: str) -> Tuple[bytes, str]:
        """
        Returns (audio_bytes, mime_type). WAV bytes for kokoro/kpipeline; pyttsx3 also returns WAV bytes.
        Raises RuntimeError with a clear explanation if nothing could be generated.
        """
        if not text:
            raise RuntimeError("TTS: empty input text.")


        # KPipeline (PyTorch)
        if self._kpipeline is not None:
            gen = self._kpipeline(text, voice=self.cfg.voice, speed=self.cfg.speed)
            chunks = []
            for _g, _p, audio_np in gen:
                if hasattr(audio_np, "detach"):
                    audio_np = audio_np.detach().cpu().numpy()
                arr = np.asarray(audio_np, dtype="float32").squeeze()
                if arr.size:
                    chunks.append(arr)
            if not chunks:
                raise RuntimeError("KPipeline returned no audio. Verify lang_code, voice, and model paths.")
            wav = np.concatenate(chunks, axis=0)
            audio_bytes = _wav_bytes_from_float32(wav, sr=24000)  # most Kokoro voices are 24 kHz
            return audio_bytes, "audio/wav"

        # pyttsx3 -> write to temp file, read bytes back
        if self._engine is not None:
            import os

            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                self._engine.save_to_file(text, wav_path)
                self._engine.runAndWait()
                # wait briefly for FS flush (Windows)
                for _ in range(20):
                    try:
                        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                            break
                    except Exception:
                        pass
                    time.sleep(0.05)
                with open(wav_path, "rb") as f:
                    data = f.read()
                if not data:
                    raise RuntimeError("pyttsx3 produced an empty WAV. Try a different voice/device.")
                return data, "audio/wav"
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        raise RuntimeError("No TTS backend initialized.")



        # if self._engine is not None:
        #     # pyttsx3 writes to file—capture to memory via temp
        #     import tempfile, wave
        #     import pyaudio  # if missing, user can switch backends
        #     # simplest: speak to device is easy; saving to file needs a driver; to keep it simple,
        #     # we’ll just return empty for pyttsx3 if not configured. (kokoro path is recommended)
        #     return b"", "audio/wav"
        #
        # return b"", "audio/wav"


# ======================
# Factory helpers
# ======================
def init_stt(stt_cfg_dict: dict) -> SpeechToText:
    cfg = STTConfig(**(stt_cfg_dict or {}))
    return SpeechToText(cfg)


def init_tts(cfg: dict) -> TextToSpeech:
    """
    Factory used by the app. Accepts aliases:
      - 'kokoro' == 'kokoro_onnx'
      - 'kokoro_pipeline' / 'kpipeline' == 'kokoro_kpipeline'
    """
    t = TTSConfig(
        backend=cfg.get("backend", "kokoro_kpipeline"),
        model_dir=cfg.get("model_dir"),
        voices_dir=cfg.get("voices_dir"),
        lang_code=cfg.get("lang_code"),
        voice=cfg.get("voice", "ff_siwis"),
        speed=float(cfg.get("speed", 1.0)),
        output_format=cfg.get("output_format", "wav"),
    )
    return TextToSpeech(t)
