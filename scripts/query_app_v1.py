#!/usr/bin/env python3
import os
import json
from typing import Any, Dict, List, Tuple, Optional
import re
import numpy as np
import faiss
import yaml

# Embeddings for query vector
from sentence_transformers import SentenceTransformer

# Optional reranker
try:
    from sentence_transformers import CrossEncoder
    _RERANK_OK = True
except Exception:
    _RERANK_OK = False

# Optional tiktoken to trim context by tokens
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

# Ollama client (local LLM)
try:
    import ollama
    _OLLAMA_OK = True
except Exception:
    _OLLAMA_OK = False

# llama-cpp client
try:
    from llama_cpp import Llama
    _LLAMACPP_OK = True
except Exception:
    _LLAMACPP_OK = False


_llm_llama = None
def get_llama_cpp(gen_cfg: Dict[str, Any]) -> "Llama":
    global _llm_llama
    if _llm_llama is not None:
        return _llm_llama
    if not _LLAMACPP_OK:
        raise RuntimeError("llama-cpp-python not installed")
    _llm_llama = Llama(
        model_path=gen_cfg["model_path"],
        n_ctx=int(gen_cfg.get("num_ctx", 4096)),
        n_gpu_layers=int(gen_cfg.get("n_gpu_layers", 0)),   # CPU
        n_threads=int(gen_cfg.get("n_threads", 8)),
        n_batch=int(gen_cfg.get("n_batch", 256)),
        logits_all=False,
        verbose=False,
    )
    return _llm_llama

def generate_with_llama_cpp(messages: List[Dict[str, str]], gen_cfg: Dict[str, Any]) -> str:
    llm = get_llama_cpp(gen_cfg)
    out = llm.create_chat_completion(
        messages=messages,
        temperature=float(gen_cfg.get("temperature", 0.2)),
        max_tokens=int(gen_cfg.get("max_tokens", 400)),
    )
    return out["choices"][0]["message"]["content"]


def read_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or os.getenv("CONFIG") or "config.yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_index(index_dir: str) -> Tuple[faiss.IndexIDMap2, List[Dict[str, Any]], Dict[str, Any]]:
    index_path = os.path.join(index_dir, "index.faiss")
    ds_path = os.path.join(index_dir, "docstore.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

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


def embed_query(query: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


def pretty_citation(md: Dict[str, Any]) -> str:
    src = md.get("source")
    if src == "genereviews":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id")
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "clinvar":
        gs = md.get("gene_symbol")
        vid = md.get("variant_id")
        return f"ClinVar: {vid or 'variant'} in {gs or 'gene'}"
    if src == "mitocarta":
        return f"Mitocarta gene: {md.get('symbol')}"
    # Legacy fallback
    if src == "pdf":
        sec = md.get("section") or "Section"
        page = md.get("page")
        title = md.get("doc_title") or md.get("doc_id")
        cite = f"{title} – {sec}"
        if page:
            cite += f", p.{page}"
        return cite
    if src == "json_variants":
        return f"Variant {md.get('variant_id')} in {md.get('gene_symbol')}"
    if src == "json_genes":
        return f"Gene {md.get('symbol')}"
    return "Source"


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
    # Map legacy keys to new ones
    if "pdf" in boosts and "genereviews" not in boosts:
        boosts["genereviews"] = boosts["pdf"]
    if "json_variants" in boosts and "clinvar" not in boosts:
        boosts["clinvar"] = boosts["json_variants"]
    if "json_genes" in boosts and "mitocarta" not in boosts:
        boosts["mitocarta"] = boosts["json_genes"]
    return boosts

def apply_source_boosts(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], prefer_genereviews: bool) -> None:
    boosts = _normalize_boosts(retr_cfg.get("source_boosts") or {})
    # If guidance-like, increase GeneReviews preference and slightly damp JSONs
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
    terms = GUIDANCE_KEYWORDS_DEFAULT
    tl = text.lower()
    count = 0
    for t in terms:
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
    # New preferred keys
    gr_min = int(caps.get("genereviews_min", caps.get("pdf_min", 3)))
    clinvar_max = int(caps.get("clinvar_max", caps.get("json_variants_max", 2)))
    mitocarta_max = int(caps.get("mitocarta_max", caps.get("json_genes_max", 2)))
    return gr_min, clinvar_max, mitocarta_max

def rebalance_by_source(hits: List[Dict[str, Any]], retr_cfg: Dict[str, Any], final_k: int) -> List[Dict[str, Any]]:
    gr_min, clinvar_max, mitocarta_max = _cap_values(retr_cfg)

    hits_sorted = sorted(
        hits,
        key=lambda x: x.get("_adj_score", x.get("rerank_score", x.get("score", 0.0))),
        reverse=True
    )

    gr = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "genereviews"]
    cl = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "clinvar"]
    mt = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") == "mitocarta"]
    other = [h for h in hits_sorted if (h.get("metadata") or {}).get("source") not in {"genereviews","clinvar","mitocarta"}]

    result = []
    used = set()

    # Ensure minimum GeneReviews
    for h in gr[:min(gr_min, len(gr))]:
        if h["idx"] not in used:
            result.append(h); used.add(h["idx"])

    # Cap ClinVar and MitoCarta
    for pool, cap in [(cl, clinvar_max), (mt, mitocarta_max)]:
        for h in pool:
            if len([x for x in result if (x.get("metadata") or {}).get("source") == (h.get("metadata") or {}).get("source")]) >= cap:
                break
            if h["idx"] not in used:
                result.append(h); used.add(h["idx"])

    # Fill remaining slots: prefer GeneReviews, then others
    def take_from(pool):
        nonlocal result, used
        for h in pool:
            if len(result) >= final_k:
                break
            if h["idx"] in used:
                continue
            result.append(h); used.add(h["idx"])

    if len(result) < final_k: take_from(gr)
    if len(result) < final_k: take_from(other)
    if len(result) < final_k: take_from(cl)
    if len(result) < final_k: take_from(mt)

    return result[:final_k]


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

def load_id_maps(index_dir: str) -> Dict[str, Any]:
    p = os.path.join(index_dir, "id_maps.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"variant_id_to_idx": {}, "rsid_to_idx": {}, "gene_symbol_to_idxs": {}}


# Patterns to extract IDs from within a sentence
VARIANT_ID_RE = re.compile(r"\b(?:(?:chr)?([0-9]{1,2}|X|Y|MT)):(\d+):([ACGT]):([ACGT])\b", re.I)
RSID_RE = re.compile(r"\brs\d+\b", re.I)

def detect_identifiers(query: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    # Variant IDs like X:155026961:C:A or chrX:... (normalize to X:155026961:C:A)
    for m in VARIANT_ID_RE.finditer(query):
        chrom = m.group(1).upper()
        pos = m.group(2)
        ref = m.group(3).upper()
        alt = m.group(4).upper()
        out.append(("variant_id", f"{chrom}:{pos}:{ref}:{alt}"))
    # rsIDs
    for m in RSID_RE.finditer(query):
        out.append(("rsid", m.group(0).lower()))
    # Gene symbol: only if clearly indicated (avoid false positives)
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
                "idx": int(i),  # unified docstore index
                "text": rec["text"],
                "metadata": rec.get("metadata", {}),
                "_adj_score": 999.0,  # force to top later
            })
            seen_idx.add(i)
    return hits


def search(
    query: str,
    index: faiss.IndexIDMap2,
    docstore: List[Dict[str, Any]],
    embed_model_name: str,
    top_k: int = 20,
    enable_reranker: bool = True,
    rerank_top_k: int = 5,
    reranker_model: str = "BAAI/bge-reranker-base",
    retr_cfg: Optional[Dict[str, Any]] = None,
    id_maps: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    retr_cfg = retr_cfg or {}

    # 0) Structured fast-path
    struct_hits = structured_lookup_first(query, id_maps or {}, docstore, limit_per_key=3)

    # 1) Dense retrieval
    qemb = embed_query(query, embed_model_name)
    D, I = index.search(qemb, top_k)
    I = I[0]; D = D[0]

    dense_hits = []
    for score, idx in zip(D, I):
        if idx == -1:
            continue
        rec = docstore[idx]
        dense_hits.append({
            "score": float(score),
            "idx": int(idx),
            "text": rec["text"],
            "metadata": rec.get("metadata", {}),
        })

    # 2) Optional reranker over dense hits
    if enable_reranker and _RERANK_OK and rerank_top_k and dense_hits:
        try:
            ce = CrossEncoder(reranker_model)
            pairs = [(query, h["text"]) for h in dense_hits]
            rr = ce.predict(pairs)
            for i, s in enumerate(rr):
                dense_hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in dense_hits:
                h["rerank_score"] = h["score"]
    else:
        for h in dense_hits:
            h["rerank_score"] = h["score"]

    # 3) Merge structured hits first, then dense (dedupe by idx)
    seen = {h["idx"] for h in struct_hits}
    hits = struct_hits + [h for h in dense_hits if h["idx"] not in seen]
    # Truncate to candidate pool before final rebalancing
    if rerank_top_k:
        hits = hits[:rerank_top_k]

    # 4) Source-aware boosts and lexical bump
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)

    # 5) Per-source quotas
    final_k = int(retr_cfg.get("final_k", rerank_top_k or len(hits)))
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits


def search_multi_stage(
    query: str,
    embed_model_name: str,
    per_source_indices: Dict[str, Tuple[Optional[faiss.IndexIDMap2], Optional[List[Dict[str, Any]]]]],
    per_source_k: Dict[str, int],
    final_k: int,
    enable_reranker: bool,
    reranker_model: str,
    retr_cfg: Dict[str, Any],
    id_maps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # 0) Structured fast-path (unified docstore isn't provided here; skip or rely on final merge)
    # We'll still do unified structured fast-path at call site if desired. Here we just dense-search subindices.

    # 1) For each subindex, retrieve k candidates
    model = SentenceTransformer(embed_model_name)
    qemb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def search_ix(ix, ds, k):
        if not ix or not ds or k <= 0:
            return []
        D, I = ix.search(qemb, k)
        D, I = D[0], I[0]
        out = []
        for score, idx in zip(D, I):
            if idx == -1: continue
            rec = ds[idx]
            out.append({
                "score": float(score),
                "idx": int(idx),
                "text": rec["text"],
                "metadata": rec.get("metadata", {}),
            })
        return out

    hits = []
    # Normalize keys (accept either genereviews/pdf, clinvar/json_variants, mitocarta/json_genes)
    key_map = {
        "genereviews": "genereviews", "pdf": "genereviews",
        "clinvar": "clinvar", "json_variants": "clinvar",
        "mitocarta": "mitocarta", "json_genes": "mitocarta",
    }
    for k_raw, k in key_map.items():
        if k_raw in per_source_k:
            ix, ds = per_source_indices.get(k, (None, None))
            hits.extend(search_ix(ix, ds, int(per_source_k[k_raw])))

    if not hits:
        return []

    # 2) Rerank across merged candidates
    if enable_reranker and _RERANK_OK:
        try:
            ce = CrossEncoder(reranker_model)
            pairs = [(query, h["text"]) for h in hits]
            rr = ce.predict(pairs)
            for i, s in enumerate(rr):
                hits[i]["rerank_score"] = float(s)
        except Exception:
            for h in hits:
                h["rerank_score"] = h["score"]
    else:
        for h in hits:
            h["rerank_score"] = h["score"]

    # 3) Boosts and lexical
    prefer_gr = classify_guidance_query(query, retr_cfg)
    apply_source_boosts(hits, retr_cfg, prefer_gr)
    add_lexical_boosts_for_genereviews(hits, query, retr_cfg)

    # 4) Final selection with per-source quotas
    hits = rebalance_by_source(hits, retr_cfg, final_k=final_k)
    return hits



def trim_to_token_budget(text: str, max_tokens: int) -> str:
    if not _enc or max_tokens is None:
        # fallback: character budget ~ 4 chars/token
        budget_chars = max_tokens * 4 if max_tokens else 8000
        return text[:budget_chars]
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])


def build_messages(query: str, hits: List[Dict[str, Any]], system_prompt: str, num_ctx: int) -> List[Dict[str, str]]:
    # Build a numbered context with citations [1], [2], ...
    context_blocks = []
    for i, h in enumerate(hits, 1):
        src = pretty_citation(h["metadata"])
        block = f"[{i}] {src}\n{h['text']}"
        context_blocks.append(block)

    # Aim to leave room for the answer; keep ~60% of context window for prompt+context
    # Adjust as needed
    ctx_budget = int(num_ctx * 0.6) if num_ctx else None
    context = "\n\n".join(context_blocks)
    if ctx_budget:
        context = trim_to_token_budget(context, ctx_budget)

    user_prompt = (
        "Use only the context to answer the question. If the answer is not present, say you don’t know.\n"
        "Cite sources in brackets like [1], [2] that refer to the numbered context blocks.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_with_ollama(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, num_ctx: int) -> str:
    if not _OLLAMA_OK:
        raise RuntimeError("ollama Python package not installed. pip install ollama")
    # Map to Ollama chat
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": float(temperature),
            "num_ctx": int(num_ctx) if num_ctx else None,
            "num_predict": int(max_tokens) if max_tokens else None,
        },
    )
    return resp["message"]["content"]


def generate_with_hf(messages: List[Dict[str, str]], gen_cfg: Dict[str, Any]) -> str:
    raise RuntimeError("HF provider not implemented in this script. Set generation.provider=llama_cpp or add a HF pipeline.")


def _hit_key(h: Dict[str, Any]) -> Tuple[Any, ...]:
    md = h.get("metadata", {}) or {}
    src = md.get("source")
    # Prefer stable identity by content, not just idx
    return (
        src,
        md.get("variant_id") or md.get("symbol") or md.get("doc_id"),
        md.get("page"),
        md.get("section"),
    )


def main(config_path: Optional[str] = None):
    cfg = read_config(config_path)
    index_dir = cfg["paths"]["index_dir"]

    # Retrieval settings
    retr = cfg.get("retrieval", {})
    top_k = int(retr.get("top_k", 20))
    enable_reranker = bool(retr.get("enable_reranker", True))
    rerank_top_k = int(retr.get("rerank_top_k", 5))
    reranker_model = retr.get("reranker_model", "BAAI/bge-reranker-base")

    # Generation settings
    gen = cfg.get("generation", {})
    provider = str(gen.get("provider", "llama_cpp")).lower()  # transformers | llama_cpp | ollama
    # Common options (some providers ignore some fields)
    temperature = float(gen.get("temperature", 0.2))
    max_tokens = int(gen.get("max_tokens", 600))
    num_ctx = int(gen.get("num_ctx", 4096))
    system_prompt = gen.get("system_prompt", "")

    # Load retrieval index and metadata
    index, docstore, meta = load_index(index_dir)
    embed_model_name = meta.get("embed_model_name") or cfg["embedding"]["model_name"]

    # Try subindices (genereviews/clinvar/mitocarta)
    gr_ix, gr_ds = load_subindex(index_dir, "genereviews")
    cl_ix, cl_ds = load_subindex(index_dir, "clinvar")
    mt_ix, mt_ds = load_subindex(index_dir, "mitocarta")
    sub = {"genereviews": (gr_ix, gr_ds), "clinvar": (cl_ix, cl_ds), "mitocarta": (mt_ix, mt_ds)}
    have_sub = any(ix is not None and ds is not None for ix, ds in sub.values())

    # Load ID maps for structured lookup (if present)
    id_maps = load_id_maps(index_dir)

    # Multi-stage retrieval settings
    per_source_k = retr.get("per_source_k", {"genereviews": 40, "clinvar": 6, "mitocarta": 6})
    final_k = int(retr.get("final_k", retr.get("rerank_top_k", 8)))
    multi_stage = bool(retr.get("multi_stage", False))

    print("Loaded index. Type a question, or 'exit' to quit.")

    while True:
        try:
            q = input("\nQuery: ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        # Retrieve
        retr_cfg = retr  # pass full retrieval section to helpers

        # Structured fast-path (exact IDs) using the unified docstore
        struct_hits = structured_lookup_first(q, id_maps, docstore, limit_per_key=3)

        # Dense retrieval
        if multi_stage and have_sub:
            dense_hits = search_multi_stage(
                q,
                embed_model_name=embed_model_name,
                per_source_indices=sub,
                per_source_k=retr.get("per_source_k", {"genereviews": 40, "clinvar": 6, "mitocarta": 6}),
                final_k=int(retr.get("final_k", retr.get("rerank_top_k", 8))),
                enable_reranker=enable_reranker,
                reranker_model=reranker_model,
                retr_cfg=retr_cfg,
                id_maps=id_maps,  # not used inside but kept for signature
            )
        else:
            dense_hits = search(
                q, index, docstore, embed_model_name,
                top_k=int(retr.get("top_k", 50)),
                enable_reranker=enable_reranker,
                rerank_top_k=int(retr.get("rerank_top_k", 8)),
                reranker_model=reranker_model,
                retr_cfg=retr_cfg,
                id_maps=id_maps,
            )

        # Merge structured hits in front, dedupe by content
        hits = []
        seen = set()
        for h in struct_hits + dense_hits:
            k = _hit_key(h)
            if k in seen:
                continue
            hits.append(h);
            seen.add(k)

        if not hits:
            print("\nNo relevant passages found. Try rephrasing your question.")
            continue


        # Show sources
        print("\nTop sources:")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {pretty_citation(h['metadata'])}")

        # Build messages for the LLM
        messages = build_messages(q, hits, system_prompt, num_ctx)

        # Generate with selected provider, then gracefully fall back to llama_cpp if available
        answer = None
        try:
            if provider == "transformers":
                # In-process HF model (4/8-bit via bitsandbytes) — configured via gen["model"]
                # Requires: transformers, accelerate, bitsandbytes
                answer = generate_with_hf(messages, gen)
            elif provider == "llama_cpp":
                # In-process GGUF via llama.cpp — configured via gen["model_path"]
                # Requires: llama-cpp-python
                answer = generate_with_llama_cpp(messages, gen)
            elif provider == "ollama":
                # Local server (may be blocked in your environment)
                # Requires: ollama package and running daemon
                model_name = gen.get("model", "qwen2.5:3b-instruct")
                answer = generate_with_ollama(messages, model_name, temperature, max_tokens, num_ctx)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            print(f"\nGeneration error with provider '{provider}': {e}")

        # Optional fallback: try llama_cpp if the primary provider failed and a GGUF is configured
        if answer is None and gen.get("model_path"):
            try:
                print("Falling back to llama_cpp (GGUF)…")
                answer = generate_with_llama_cpp(messages, gen)
            except Exception as e2:
                print(f"Fallback (llama_cpp) also failed: {e2}")

        if answer is None:
            print("\nCould not generate an answer. Please verify your 'generation' config and installed packages.")
            continue

        print("\nAnswer:\n")
        print(answer)

        # Append explicit source list for user verification
        print("\nSources:")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {pretty_citation(h['metadata'])}")


if __name__ == "__main__":
    main()
