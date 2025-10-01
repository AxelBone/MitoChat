#!/usr/bin/env python3
import os
import json
from typing import Any, Dict, List, Tuple, Optional

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


def search(
    query: str,
    index: faiss.IndexIDMap2,
    docstore: List[Dict[str, Any]],
    embed_model_name: str,
    top_k: int = 20,
    enable_reranker: bool = True,
    rerank_top_k: int = 5,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> List[Dict[str, Any]]:
    qemb = embed_query(query, embed_model_name)
    D, I = index.search(qemb, top_k)
    I = I[0]; D = D[0]

    hits = []
    for score, idx in zip(D, I):
        if idx == -1:
            continue
        rec = docstore[idx]
        hits.append({
            "score": float(score),
            "idx": int(idx),
            "text": rec["text"],
            "metadata": rec["metadata"],
        })

    if enable_reranker and _RERANK_OK and rerank_top_k and hits:
        ce = CrossEncoder(reranker_model)
        pairs = [(query, h["text"]) for h in hits]
        rr = ce.predict(pairs)
        for i, s in enumerate(rr):
            hits[i]["rerank_score"] = float(s)
        hits.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        hits = hits[:rerank_top_k]
    else:
        hits = hits[:rerank_top_k if rerank_top_k else len(hits)]
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
    provider = str(gen.get("provider", "transformers")).lower()  # transformers | llama_cpp | ollama
    # Common options (some providers ignore some fields)
    temperature = float(gen.get("temperature", 0.2))
    max_tokens = int(gen.get("max_tokens", 600))
    num_ctx = int(gen.get("num_ctx", 4096))
    system_prompt = gen.get("system_prompt", "")

    # Load retrieval index and metadata
    index, docstore, meta = load_index(index_dir)
    embed_model_name = meta.get("embed_model_name") or cfg["embedding"]["model_name"]
    print("Loaded index. Type a question, or 'exit' to quit.")

    while True:
        try:
            q = input("\nQuery: ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        # Retrieve
        hits = search(
            q, index, docstore, embed_model_name,
            top_k=top_k, enable_reranker=enable_reranker,
            rerank_top_k=rerank_top_k, reranker_model=reranker_model
        )

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
