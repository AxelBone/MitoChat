#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import re
import json
import math
import uuid
import yaml
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer

# Tokenizer (optional)
try:
    import tiktoken
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_enc = None

# FAISS
import faiss

# Optional PDF processing via Unstructured; fallback to PyMuPDF or plain text
_UNSTRUCTURED_OK = True
try:
    from unstructured.partition.pdf import partition_pdf
except Exception:
    _UNSTRUCTURED_OK = False

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


ALLOWED_PDF_EXT = {".pdf", ".txt"}
ALLOWED_JSON_EXT = {".json"}

def expand_to_files(spec, allowed_exts) -> List[str]:
    """
    Accepts:
      - None
      - a single string: file path, dir path, or glob pattern
      - a list of strings (mix allowed)
    Returns a sorted list of file paths filtered by allowed_exts.
    """
    if not spec:
        return []
    if isinstance(spec, str):
        specs = [spec]
    else:
        specs = list(spec)

    files = []
    for s in specs:
        p = Path(s)
        if any(ch in s for ch in "*?[]"):  # glob
            for g in glob.glob(s):
                gp = Path(g)
                if gp.is_file() and gp.suffix.lower() in allowed_exts:
                    files.append(str(gp))
        elif p.is_dir():
            for child in p.iterdir():
                if child.is_file() and child.suffix.lower() in allowed_exts:
                    files.append(str(child))
        elif p.is_file():
            if p.suffix.lower() in allowed_exts:
                files.append(str(p))
        else:
            # ignore missing paths; you can print a notice if desired
            pass
    # Stable order
    files = sorted(set(files))
    return files


def expand_inputs_from_cfg(cfg: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Prefer new multi-input keys, fall back to legacy single-path keys for compatibility.
    """
    paths = cfg.get("paths", {})
    pdf_files = expand_to_files(paths.get("genereviews_inputs"), ALLOWED_PDF_EXT)
    var_files = expand_to_files(paths.get("clinvar_inputs"), ALLOWED_JSON_EXT)
    gene_files = expand_to_files(paths.get("mitocarta_inputs"), ALLOWED_JSON_EXT)

    # Legacy single-file support
    if not pdf_files and paths.get("pdf_path"):
        pdf_files = [paths["pdf_path"]]
    if not var_files and paths.get("variants_path"):
        var_files = [paths["variants_path"]]
    if not gene_files and paths.get("genes_path"):
        gene_files = [paths["genes_path"]]
    return pdf_files, var_files, gene_files


def read_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    # Allow env override if configured
    default_path = "config.yaml"
    path = config_path or os.getenv("CONFIG") or default_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_tokens(text: str) -> int:
    if _tiktoken_enc is not None:
        return len(_tiktoken_enc.encode(text))
    return max(1, len(text.split()))


def split_text(text: str, max_tokens: int = 700, overlap: int = 100) -> List[str]:
    if not text.strip():
        return []
    if _tiktoken_enc is None:
        words = text.split()
        step = max(1, max_tokens - overlap)
        size = max(1, max_tokens)
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i:i+size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
        return chunks

    toks = _tiktoken_enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        chunk = _tiktoken_enc.decode(window)
        chunks.append(chunk)
        i += max_tokens - overlap
    return chunks


def clean_decimal_string(s: str) -> Optional[float]:
    """
    Converts strings like '43,6442' or '2,05E+11' (with comma decimals or sci notation) to float.
    Leaves plain integers like '34' alone by returning None so caller can decide.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    # Only convert if it looks like a decimal with comma/dot or scientific notation
    has_comma_decimal = ("," in s and any(ch.isdigit() for ch in s))
    sci_notation = bool(re.search(r"^[+-]?\d+(?:[.,]\d+)?[eE][+-]?\d+$", s))
    decimal_point = ("." in s and any(ch.isdigit() for ch in s) and not s.isdigit())
    if not (has_comma_decimal or sci_notation or decimal_point):
        return None
    try:
        s2 = s.replace(",", ".")
        return float(s2)
    except Exception:
        return None


def infer_genome_build_from_hgvs(hgvs: str) -> Optional[str]:
    if not hgvs:
        return None
    if "NC_012920" in hgvs:
        return "rCRS"
    if re.search(r"NC_0000\d{2}\.1[1-9]", hgvs):
        return "GRCh38"
    return None


def normalize_clnsig(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    canon = label.strip().lower().replace(" ", "_")
    mapping = {
        "benign/likely_benign": "benign",
        "likely_benign": "likely_benign",
        "benign": "benign",
        "uncertain_significance": "vus",
        "likely_pathogenic": "likely_pathogenic",
        "pathogenic": "pathogenic",
    }
    return mapping.get(canon, canon)


def ensure_rsid(rs: Optional[str]) -> Optional[str]:
    if rs is None or str(rs).strip().lower() in {"", "nan", "none"}:
        return None
    s = str(rs).strip()
    if s.startswith("rs"):
        return s
    if s.isdigit():
        return f"rs{s}"
    return s


def flatten_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v not in (None, "", "NaN", "nan")]
    return [str(x)]


def load_json_relaxed(path: str) -> Any:
    """
    Loads JSON allowing bare NaN/Infinity tokens by replacing with null before parsing.
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # Replace unquoted NaN/Infinity with null
    txt = re.sub(r'(?<![A-Za-z0-9_])NaN(?![A-Za-z0-9_])', 'null', txt)
    txt = re.sub(r'(?<![A-Za-z0-9_\-])Infinity(?![A-Za-z0-9_])', 'null', txt)
    txt = re.sub(r'(?<![A-Za-z0-9_\-])-Infinity(?![A-Za-z0-9_])', 'null', txt)
    return json.loads(txt)


def parse_pdf_with_unstructured(pdf_path: str) -> List[Dict[str, Any]]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        include_page_breaks=True,
    )
    out = []
    for el in elements:
        txt = (getattr(el, "text", "") or "").strip()
        if not txt:
            continue
        meta = getattr(el, "metadata", None)
        page_no = getattr(meta, "page_number", None) if meta else None
        category = getattr(el, "category", None)
        out.append({
            "text": txt,
            "page_number": page_no,
            "category": category or "Unknown",
        })
    return out


def parse_pdf_with_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed; cannot parse PDF fallback.")
    doc = fitz.open(pdf_path)
    elems = []
    for i in range(len(doc)):
        page = doc[i]
        txt = page.get_text("text")
        if not txt:
            continue
        blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
        for b in blocks:
            cat = "Title" if re.match(r"^[A-Z0-9][A-Z0-9 \-\(\):]{5,}$", b) else "NarrativeText"
            elems.append({"text": b, "page_number": i+1, "category": cat})
    return elems


def parse_plain_text(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    blocks = [b.strip() for b in re.split(r"\n{2,}", txt) if b.strip()]
    elems = []
    page_no = None
    for b in blocks:
        cat = "Title" if re.match(r"^[A-Z0-9][A-Z0-9 \-\(\):]{5,}$", b) else "NarrativeText"
        elems.append({"text": b, "page_number": page_no, "category": cat})
    return elems


def make_pdf_chunks(
    pdf_elements: List[Dict[str, Any]],
    doc_id: str,
    doc_title: Optional[str],
    max_tokens: int = 700,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    chunks = []
    current_section = None
    for el in pdf_elements:
        cat = (el.get("category") or "NarrativeText").lower()
        text = el["text"]
        page = el.get("page_number")

        if cat == "title":
            current_section = text[:200]
            continue

        if cat == "table":
            meta = {
                "source": "pdf",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section": current_section,
                "page": page,
                "block_type": "table",
            }
            chunks.append({"id": str(uuid.uuid4()), "text": text, "metadata": meta})
            continue

        parts = split_text(text, max_tokens=max_tokens, overlap=overlap)
        for p in parts:
            meta = {
                "source": "pdf",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section": current_section,
                "page": page,
                "block_type": "text",
            }
            chunks.append({"id": str(uuid.uuid4()), "text": p, "metadata": meta})
    return chunks


def make_variant_records(variants_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for key, rec in variants_json.items():
        info = rec.get("info", {}) or {}
        clnhgvs = info.get("CLNHGVS")
        build = infer_genome_build_from_hgvs(clnhgvs or "")
        clnsig_raw = info.get("CLNSIG")
        clnsig = normalize_clnsig(clnsig_raw)
        mc = info.get("MC")
        consequence = None
        if isinstance(mc, str) and "|" in mc:
            consequence = mc.split("|", 1)[1]
        rsid = ensure_rsid(rec.get("rsid"))
        af_exac = info.get("AF_EXAC")
        af_exac_v = None
        if af_exac is not None:
            try:
                af_exac_v = float(af_exac)
            except Exception:
                af_exac_v = clean_decimal_string(str(af_exac))

        diseases = []
        clndn = info.get("CLNDN")
        if clndn:
            diseases = [d.strip().replace("_", " ") for d in str(clndn).split("|") if d.strip()]

        text_view = (
            f"Variant {rec.get('variant_id')} in {rec.get('gene_symbol')} "
            f"({clnhgvs or 'HGVS N/A'}). ClinVar significance: {clnsig or clnsig_raw}. "
            f"Diseases: {', '.join(diseases) or 'N/A'}. Consequence: {consequence or 'N/A'}. "
            f"rsID: {rsid or 'N/A'}. Build: {build or 'unknown'}. "
            f"Review: {info.get('CLNREVSTAT') or 'N/A'}."
        )

        position = rec.get("position")
        try:
            position = int(position)
        except Exception:
            pass

        meta = {
            "source": "json_variants",
            "json_key": key,
            "variant_id": rec.get("variant_id"),
            "gene_symbol": rec.get("gene_symbol"),
            "gene_id": rec.get("gene_id"),
            "chromosome": rec.get("chromosome"),
            "position": position,
            "ref": rec.get("ref"),
            "alt": rec.get("alt"),
            "clnhgvs": clnhgvs,
            "genome_build": build,
            "clnsig": clnsig or clnsig_raw,
            "consequence": consequence,
            "diseases": diseases,
            "rsid": rsid,
            "review_status": info.get("CLNREVSTAT"),
        }

        out.append({"id": str(uuid.uuid4()), "text": text_view, "metadata": meta})
    return out


def clean_numbers_in_obj(obj: Any) -> Any:
    """
    Recursively normalizes only decimal/sci-notation strings to floats; preserves plain integer-like strings.
    Replaces 'NaN'/'Infinity' with None.
    """
    if obj is None:
        return None
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_numbers_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_numbers_in_obj(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s.lower() in {"nan", "none", ""}:
            return None
        # Try decimal/sci conversion only
        val = clean_decimal_string(s)
        return val if val is not None else obj
    return obj


def make_gene_records(genes_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    genes_json = clean_numbers_in_obj(genes_json)
    out = []
    for key, rec in genes_json.items():
        symbol = rec.get("symbol")
        desc = rec.get("description")

        # Force-cast to str and drop '-' placeholders
        syns_raw = rec.get("synonyms") or []
        syns = [str(s) for s in syns_raw if s not in (None, "-", "")]

        mc3 = rec.get("mitocarta3") or {}
        sublocs = mc3.get("sub_mito_localization") or []
        pathways = mc3.get("mito_pathways") or []
        hpa_loc = rec.get("hpa_location_2020")
        coords = rec.get("genome_coords_hg19") or {}
        chrom = coords.get("chromosome")
        start = coords.get("start")
        stop = coords.get("stop")

        parts = []
        parts.append(f"{symbol}: {desc}" if desc else f"{symbol or ''}".strip())
        if sublocs:
            parts.append(f"Localization: {', '.join(map(str, sublocs))}")
        if pathways:
            parts.append(f"Pathways: {', '.join(map(str, pathways[:6]))}{'...' if len(pathways) > 6 else ''}")
        if syns:
            parts.append(f"Synonyms: {', '.join(syns[:8])}{'...' if len(syns) > 8 else ''}")
        if hpa_loc:
            parts.append(f"HPA location: {hpa_loc}")
        if chrom and start and stop:
            try:
                parts.append(f"hg19 coords: {chrom}:{int(start)}-{int(stop)}")
            except Exception:
                parts.append(f"hg19 coords: {chrom}:{start}-{stop}")

        text_view = ". ".join([p for p in parts if p])

        def to_int(v):
            try:
                return int(v)
            except Exception:
                return None

        meta = {
            "source": "json_genes",
            "json_key": key,
            "symbol": symbol,
            "ensembl_id": rec.get("ensembl_id"),
            "uniprot_ids": rec.get("uniprot_ids"),
            "synonyms": syns,
            "sub_mito_localization": sublocs,
            "mito_pathways": pathways,
            "hpa_location_2020": hpa_loc,
            "genome_build": "hg19",
            "chromosome": chrom,
            "start": to_int(start),
            "stop": to_int(stop),
        }

        out.append({"id": str(uuid.uuid4()), "text": text_view, "metadata": meta})
    return out


def build_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> Tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model_name})"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=normalize)
        vecs.append(embs)
    X = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")
    return X.astype("float32"), model


def build_faiss_index(embeddings: np.ndarray, ids: List[int]) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    id_index = faiss.IndexIDMap2(index)
    id_arr = np.array(ids, dtype=np.int64)
    id_index.add_with_ids(embeddings, id_arr)
    return id_index


def save_index(index: faiss.IndexIDMap2, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))


def save_docstore(records: List[Dict[str, Any]], index_dir: str):
    path = os.path.join(index_dir, "docstore.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_pdf_elements(path: str, prefer_unstructured: bool) -> Tuple[List[Dict[str, Any]], str]:
    doc_title = None
    if path.lower().endswith(".pdf"):
        if prefer_unstructured and _UNSTRUCTURED_OK:
            elems = parse_pdf_with_unstructured(path)
        else:
            elems = parse_pdf_with_pymupdf(path)
        for el in elems:
            if el.get("category") == "Title":
                doc_title = el.get("text")
                break
        return elems, doc_title
    else:
        elems = parse_plain_text(path)
        if elems and re.match(r"^[A-Z][A-Za-z0-9 \-\(\):]{5,}$", elems[0]["text"]):
            doc_title = elems[0]["text"]
        return elems, doc_title


def main(config_path: Optional[str] = None):
    cfg = read_config(config_path)

    # Ingestion / embedding params
    chunk_tokens = int(cfg["ingestion"]["chunk_tokens"])
    chunk_overlap = int(cfg["ingestion"]["chunk_overlap"])
    use_unstructured = bool(cfg["ingestion"]["use_unstructured"])

    embed_model_name = cfg["embedding"]["model_name"]
    embed_batch = int(cfg["embedding"]["batch_size"])
    embed_normalize = bool(cfg["embedding"]["normalize"])

    pdf_meta_note = (cfg.get("pdf_metadata") or {}).get("license_note")
    pdf_meta_url = (cfg.get("pdf_metadata") or {}).get("source_url")

    index_dir = cfg["paths"]["index_dir"]

    # 0) Expand inputs
    pdf_files, variant_json_files, gene_json_files = expand_inputs_from_cfg(cfg)

    if not pdf_files and not variant_json_files and not gene_json_files:
        raise RuntimeError("No input files found. Check paths.pdf_inputs / variants_inputs / genes_inputs in config.yaml")

    corpus: List[Dict[str, Any]] = []

    # 1) PDFs / TXTs
    if pdf_files:
        print(f"Parsing {len(pdf_files)} document(s) (PDF/TXT)...")
    for pdf_path in pdf_files:
        try:
            pdf_elements, doc_title = load_pdf_elements(pdf_path, prefer_unstructured=use_unstructured)
            doc_id = os.path.basename(pdf_path)
            pdf_chunks = make_pdf_chunks(
                pdf_elements, doc_id=doc_id, doc_title=doc_title,
                max_tokens=chunk_tokens, overlap=chunk_overlap
            )
            # tag license/source + source_file
            for ch in pdf_chunks:
                ch["metadata"]["license"] = pdf_meta_note
                ch["metadata"]["source_url"] = pdf_meta_url
                ch["metadata"]["source_note"] = "Include citation and URL when displaying content."
                ch["metadata"]["source_file"] = os.path.abspath(pdf_path)
            corpus.extend(pdf_chunks)
        except Exception as e:
            print(f"[WARN] Failed to parse {pdf_path}: {e}")

    # 2) Variants JSONs (merge dictionaries)
    if variant_json_files:
        print(f"Loading {len(variant_json_files)} variants JSON file(s)...")
    seen_var_keys = set()
    for jpath in variant_json_files:
        try:
            data = load_json_relaxed(jpath)
            if not isinstance(data, dict):
                print(f"[WARN] {jpath} is not a dict; skipping.")
                continue
            for key, rec in data.items():
                if key in seen_var_keys:
                    continue  # simple dedup by key; remove if you want to keep duplicates
                seen_var_keys.add(key)
            # Build records in one pass for this file to tag source_file properly
            file_records = make_variant_records(data)
            # Tag with source_file (and keep original json_key in metadata)
            for r in file_records:
                r["metadata"]["source_file"] = os.path.abspath(jpath)
            corpus.extend(file_records)
        except Exception as e:
            print(f"[WARN] Failed to load variants from {jpath}: {e}")

    # 3) Genes JSONs (merge dictionaries)
    if gene_json_files:
        print(f"Loading {len(gene_json_files)} genes JSON file(s)...")
    seen_gene_keys = set()
    for jpath in gene_json_files:
        try:
            data = load_json_relaxed(jpath)
            if not isinstance(data, dict):
                print(f"[WARN] {jpath} is not a dict; skipping.")
                continue
            for key in data.keys():
                if key in seen_gene_keys:
                    continue
                seen_gene_keys.add(key)
            file_records = make_gene_records(data)
            for r in file_records:
                r["metadata"]["source_file"] = os.path.abspath(jpath)
            corpus.extend(file_records)
        except Exception as e:
            print(f"[WARN] Failed to load genes from {jpath}: {e}")

    print(f"Total chunks/records: {len(corpus)}")

    # 4) Embeddings
    texts = [r["text"] for r in corpus]
    X, model = build_embeddings(
        texts, model_name=embed_model_name, batch_size=embed_batch, normalize=embed_normalize
    )

    # 5) FAISS
    print("Building FAISS index...")
    ids = list(range(len(corpus)))
    index = build_faiss_index(X, ids)

    # 6) Persist
    save_index(index, index_dir)
    save_docstore(corpus, index_dir)
    meta = {
        "embed_model_class": model.__class__.__name__,
        "embed_model_name": embed_model_name,  # from config
        "sentence_embedding_dimension": int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else None,
        "normalize": embed_normalize,
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "total_chunks": len(corpus),
        "inputs": {
            "pdf_files": pdf_files,
            "variant_json_files": variant_json_files,
            "gene_json_files": gene_json_files,
        },
    }
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Done. Index written to: {index_dir}")


    # 4) Embeddings
    texts = [r["text"] for r in corpus]
    X, model = build_embeddings(texts, model_name=embed_model_name, batch_size=embed_batch, normalize=embed_normalize)

    # 5) FAISS
    print("Building FAISS index...")
    ids = list(range(len(corpus)))
    index = build_faiss_index(X, ids)

    # 6) Persist
    save_index(index, index_dir)
    save_docstore(corpus, index_dir)
    # safer meta: do not introspect model.name_or_path (not present on some versions)
    embed_dim = int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else None
    meta = {
        "embed_model_class": model.__class__.__name__,
        "embed_model_name": embed_model_name,  # from config.yaml
        "sentence_embedding_dimension": embed_dim,
        "normalize": embed_normalize,
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "total_chunks": len(corpus),
    }

    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Done. Index written to: {index_dir}")


if __name__ == "__main__":
    # No CLI args needed. Optionally set CONFIG env var to choose a different file.
    main()
