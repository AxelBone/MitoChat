"""Streamlit app with multiple messages session
V3: Adding translation pipeline (query FR -> query EN -> processing -> answer generation EN -> answer FR)"""


import os
import json
import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import yaml
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from translate import get_translator, translate_text


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
                "trust_remote_code": gen.get("trust_remote_code", False),
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


rewrite_prompts = load_rewrite_prompts(lang="en", base_dir="../prompts")

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
    """
    meta = (hit.get("metadata") or {})
    # 1) Prefer full content already stored in metadata (if your indexer put it there)
    if meta.get("section_text"):         # full section text for PDF
        return "text", str(meta["section_text"])
    if meta.get("json_entry"):           # pre-parsed JSON entry
        return "json", meta["json_entry"]
    if meta.get("raw_json"):             # raw JSON string
        try:
            return "json", json.loads(meta["raw_json"])
        except Exception:
            return "text", str(meta["raw_json"])

    # 2) Try docstore lookups (adapt keys to your docstore schema)
    doc_id = meta.get("doc_id") or meta.get("document_id") or meta.get("source_id")
    if doc_id and docstore is not None:
        # JSON record
        rec_id = meta.get("record_id") or meta.get("entry_id") or meta.get("key")
        if rec_id is not None:
            val = docstore.get(("json", doc_id, rec_id)) or docstore.get(f"{doc_id}:{rec_id}")
            if val is not None:
                if isinstance(val, (dict, list)):
                    return "json", val
                try:
                    return "json", json.loads(val)
                except Exception:
                    return "text", str(val)

        # PDF section
        section_id = meta.get("section_id") or meta.get("sec_id")
        if section_id is not None:
            val = docstore.get(("pdf_section", doc_id, section_id)) or docstore.get(f"{doc_id}:{section_id}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

        # PDF page fallback
        page = meta.get("page") or meta.get("page_index")
        if page is not None:
            try:
                p = int(page)
            except Exception:
                p = page
            val = docstore.get(("pdf_page", doc_id, p)) or docstore.get(f"{doc_id}:p{p}")
            if val is not None:
                if isinstance(val, dict) and "text" in val:
                    return "text", str(val["text"])
                return "text", str(val)

    # 3) Final fallback: the hit chunk text itself
    return "text", str(hit.get("text", ""))


def render_sources_panel(hits: List[Dict[str, Any]], docstore: Any, expanded: bool = False):
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})", expanded=expanded):
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", {}) or {}
            title = pretty_citation(meta) or meta.get("title") or meta.get("source") or f"Source {i}"
            url = meta.get("url") or meta.get("source_url")

            st.markdown(f"({i}) {title}")
            if url:
                st.caption(url)

            kind, content = resolve_full_chunk(h, docstore)
            if kind == "json":
                if isinstance(content, (dict, list)):
                    st.json(content)
                else:
                    # content is a JSON string or other; attempt to pretty print
                    try:
                        st.json(json.loads(content))
                    except Exception:
                        st.code(str(content), language="json")
            else:
                # Full text (no preview/truncation)
                st.write(content)

            st.divider()



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



# ----------------------------
# App
# ----------------------------
def main():
    st.set_page_config(page_title="Local RAG Q&A", layout="wide")

    # Hardcoded relative path
    logo_path = "../assets/logo.png"

    col1, col2 = st.columns([1, 8], vertical_alignment="center")
    with col1:
        st.image(logo_path, use_container_width=True)
    with col2:
        st.title("Mito Chat")

    cfg = read_config()

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
        provider=trans_cfg.get("provider", "m2m"),
        fr_en_model=trans_cfg.get("fr_en_model", "Helsinki-NLP/opus-mt-fr-en"),
        en_fr_model=trans_cfg.get("en_fr_model", "Helsinki-NLP/opus-mt-en-fr"),
        model_name=trans_cfg.get("model_name", "facebook/m2m100_418M"),
        device=trans_cfg.get("device"),
        max_new_tokens=int(trans_cfg.get("max_new_tokens", 512)),
        cache_bust="m2m-cpu-v3",  # <-- change this string when you change translate.py
    )

    # Sidebar settings
    with st.sidebar:
        # Small inline action just above the input bar
        if st.button("Nouveau chat", key="btn_new_chat_inline", help="Nouvelle conversation",
                     use_container_width=True):
            st.session_state.chat = []
            st.rerun()

        st.caption(f"Compute device: {pick_device()}")

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

        # Query rewritting
        st.divider()
        use_rewrite = st.checkbox("Use query rewriting", value=True)
        show_debug = st.checkbox("Show rewrite/debug info", value=True)
        st.session_state["use_rewrite"] = use_rewrite
        st.session_state["show_debug"] = show_debug

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

    # Display existing messages (including sources for assistant turns)
    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            # Optional: show rewrite for user turns
            # if m["role"] == "user" and st.session_state.get("show_debug", True) and m.get("rewrite") and m["rewrite"] != \
            #         m["content"]:
            if m["role"] == "user" and st.session_state.get("show_debug", True) and m.get("rewrite"):
                with st.expander("Requête utilisée pour la recherche", expanded=False):
                    st.code(m["rewrite"], language="text")
            # if m.get("rewrite_input_en"):
            #     with st.expander("Traduction FR→EN (entrée du réécrivain)", expanded=False):
            #         st.code(m["rewrite_input_en"], language="text")
            #         if m.get("rewrite_debug"):
            #             with st.expander("Debug réécriture", expanded=False): st.json(m["rewrite_debug"])

            if m["role"] == "assistant" and m.get("hits"):
                with st.expander(f"Sources ({len(m['hits'])})", expanded=False):
                    display_sources = min(len(m["hits"]), 8)
                    for j, h in enumerate(m["hits"][:display_sources], 1):
                        md = h.get("metadata", {})
                        label = f"[{j}] {pretty_citation(md)}"
                        with st.expander(label, expanded=False):
                            txt = h.get("text") or ""

                            # Build a stable, unique key for this text_area
                            key_suffix = f"{i}_{j}_{md.get('source', '')}_{md.get('doc_id', '')}_{md.get('page', '')}"
                            ta_key = f"chunk_{key_suffix}"

                            st.text_area(f"Chunk {j}", txt[:50_000], height=250, key=ta_key)

                            st.markdown("---")
                            render_source_preview(md, key_suffix=key_suffix)

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
        with st.spinner("Retrieving…"):
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

        # 5) Build messages for generation (in English)
        # Tip: set your system_prompt in English in the sidebar for best results.
        messages = build_messages_with_history(standalone_q, hits, system_prompt, num_ctx, st.session_state.chat)

        # 6) Generate in EN, translate to FR, display FR, and attach EN rewrite for debug
        with st.chat_message("assistant"):
            if not hits:
                msg_en = "I couldn't find relevant passages. Try rephrasing."
                try:
                    msg_fr = translate_text(msg_en, "en", "fr", translator)
                except Exception:
                    msg_fr = "Je ne trouve pas de passages pertinents. Essayez de reformuler."
                st.markdown(msg_fr)
                st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite shown later in expander
                st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
                st.session_state.chat.append({"role": "assistant", "content": msg_fr, "hits": []})
                st.rerun()

            if provider == "llama_cpp":
                llm = get_llama_cpp_model(
                    model_path=model_path,
                    num_ctx=int(num_ctx),
                    n_gpu_layers=int(n_gpu_layers),
                    n_threads=int(n_threads),
                    n_batch=int(n_batch),
                    verbose=False,
                )
                # Buffer English; show French only at the end
                full_text_en = ""
                try:
                    for token in llm.create_chat_completion(messages=messages, temperature=float(temperature),
                                                            max_tokens=int(max_tokens), stream=True):
                        delta = token["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_text_en += delta
                except Exception:
                    out = llm.create_chat_completion(messages=messages, temperature=float(temperature),
                                                     max_tokens=int(max_tokens))
                    full_text_en = out["choices"][0]["message"]["content"]

                # Translate EN -> FR for display
                try:
                    full_text_fr = translate_text(full_text_en or "", "en", "fr", translator)
                except Exception:
                    full_text_fr = full_text_en or "Échec de traduction/génération."

                st.markdown(full_text_fr)
                st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite
                st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
                st.session_state.chat.append({"role": "assistant", "content": full_text_fr, "hits": hits})

            else:
                answer_en = generate_with_hf(messages, {
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "model": hf_model,
                    "quantization": quant,
                    "device_map": device_map,
                    "trust_remote_code": gen.get("trust_remote_code", False),
                })
                try:
                    answer_fr = translate_text(answer_en or "", "en", "fr", translator)
                except Exception:
                    answer_fr = answer_en or "Échec de génération."
                st.markdown(answer_fr)

                st.session_state.chat[user_idx]["rewrite"] = standalone_q  # English rewrite
                st.session_state.chat[user_idx]["rewrite_debug"] = rw_debug
                st.session_state.chat.append({"role": "assistant", "content": answer_fr or "", "hits": hits})

        st.rerun()


if __name__ == "__main__":
    main()
