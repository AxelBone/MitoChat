"""
mitocarta_auto_generate.py
---------------------------
Génération automatique de questions unitaires à partir de MitoCarta3.0
"""

from __future__ import annotations
import json
import random
import os
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import re


# ===============================
# CONFIGURATION
# ===============================

DATA_DIR = Path("data")
MITOCARTA_PATH = DATA_DIR / "mitocarta" / "Human.MitoCarta3.0.csv"
CHUNKS_PATH = DATA_DIR / "chunks_generated" / "chunks.jsonl"
OUTPUT_PATH = DATA_DIR / "annotations" / "questions_annotations_mitocarta.json"

SOURCE_DATASET = "mitocarta3"
PROFILE = "professionnel"
STATUS = "annoté"
DIFFICULTY = "facile"


# ===============================
# UTILITAIRES
# ===============================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def random_hex(n=8):
    return os.urandom(n // 2).hex()

def read_json_array(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

def write_json_array(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===============================
# CHARGEMENT DES DONNÉES
# ===============================

def load_mitocarta(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=None)
    print("✅ Colonnes MitoCarta3.0 chargées :", df.columns)
    return df

def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Normalisation générique
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
def match_chunks_by_term(
    chunks: list,
    term,
    source: str = "mitocarta",
    category: str | None = None
) -> list[int]:
    """Version robuste : évite les faux positifs type ETFB → ETFBKMT."""
    if not term:
        return []

    if isinstance(term, str):
        terms = [term]
    elif isinstance(term, list):
        terms = [t for t in term if isinstance(t, str) and t.strip()]
    else:
        return []

    terms_norm = [normalize_text(t) for t in terms]
    matched = []

    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        src = str(meta.get("source", "")).lower()
        if source and source.lower() not in src:
            continue

        # ——————————————————————————————
        # 🧩 Définition des champs pertinents selon la catégorie
        # ——————————————————————————————
        if category == "gene":
            haystack_fields = [
                meta.get("symbol", ""),
                meta.get("ensembl_id", ""),
                *(meta.get("uniprot_ids", []) or []),
                *(meta.get("synonyms", []) or []),
            ]
        elif category == "localization":
            haystack_fields = [
                *(meta.get("sub_mito_localization", []) or []),
                meta.get("hpa_location_2020", "")
            ]
        elif category == "pathway":
            haystack_fields = meta.get("mito_pathways", []) or []
        else:
            # fallback générique
            haystack_fields = [ch.get("text", "")]

        # ——————————————————————————————
        # 🔎 Recherche stricte par mot ou par champ complet
        # ——————————————————————————————
        fields_norm = [normalize_text(f) for f in haystack_fields]
        if any(t == f for t in terms_norm for f in fields_norm):
            matched.append(ch.get("idx"))
            continue

        # 🩹 Fallback secondaire : recherche par mots entiers dans le texte
        text_norm = normalize_text(ch.get("text", ""))
        for t in terms_norm:
            if re.search(rf"\b{re.escape(t)}\b", text_norm):
                matched.append(ch.get("idx"))
                break

    return sorted(set(filter(None, matched)))




# ===============================
# 1️⃣ ENSEMBL ID → GENE SYMBOL
# ===============================

def template_ensembl_to_gene(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    df_valid = df.drop_nulls(subset=["EnsemblGeneID_mapping_version_20200130", "Symbol"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        ensembl = row["EnsemblGeneID_mapping_version_20200130"]
        gene = row["Symbol"]
        chunk_ids = match_chunks_by_term(chunks, ensembl, source="mitocarta", category="gene")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quel est le symbole du gène humain correspondant à l’identifiant Ensembl {ensembl} ?",
            "ground_truth": f"Le symbole du gène correspondant à {ensembl} est {gene}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["ensembl_to_gene"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# 2️⃣ GENE → SUBMITO LOCALIZATION
# ===============================

def template_gene_to_localization(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    df_valid = df.drop_nulls(subset=["Symbol", "MitoCarta3.0_SubMitoLocalization"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        gene = row["Symbol"]
        loc = row["MitoCarta3.0_SubMitoLocalization"]
        loc_list = [l.strip() for l in loc.split("|") if l.strip()]
        chunk_ids = match_chunks_by_term(chunks, gene, source="mitocarta", category="gene")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quelle est la localisation sub-mitochondriale du gène {gene} ?",
            "ground_truth": f"Le gène {gene} est localisé dans {', '.join(loc_list)}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_localization"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# 3️⃣ GENE → EVIDENCE
# ===============================

def template_gene_to_evidence(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    df_valid = df.drop_nulls(subset=["Symbol", "MitoCarta3.0_Evidence"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        gene = row["Symbol"]
        evidence = [e.strip() for e in row["MitoCarta3.0_Evidence"].split(",") if e.strip()]
        chunk_ids = match_chunks_by_term(chunks, gene, source="mitocarta", category="gene")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quelles preuves expérimentales soutiennent l’inclusion du gène {gene} dans MitoCarta ?",
            "ground_truth": f"L’inclusion du gène {gene} dans MitoCarta3.0 est soutenue par {', '.join(evidence)}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_evidence"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# 4️⃣ GENE → PROTEIN LENGTH
# ===============================

def template_gene_to_proteinlength(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    df_valid = df.drop_nulls(subset=["Symbol", "ProteinLength"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        gene = row["Symbol"]
        length = row["ProteinLength"]
        chunk_ids = match_chunks_by_term(chunks, gene, source="mitocarta", category="gene")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quelle est la longueur de la protéine codée par le gène {gene} ?",
            "ground_truth": f"Le gène {gene} code pour une protéine de {length} acides aminés.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_proteinlength"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# 5️⃣ GENE → TISSUES
# ===============================

def format_tissue_answer(gene: str, tissues_raw: list[str]) -> tuple[str, str]:
    """
    Adapte la réponse selon la valeur du champ Tissues :
    - all_14 / all 14 → ubiquitaire
    - none / not_detected → absente
    - cas normal : liste réelle de tissus
    """
    # Normalisation et nettoyage
    tissues = [t.strip() for t in tissues_raw if t.strip()]
    joined = " ".join(tissues).lower()  # pour analyse sémantique rapide
    n_tissues = len(tissues)

    # 🔍 Détection symbolique
    if any(x in joined for x in ["all_14", "all 14", "14 tissues", "all tissues"]):
        query = f"Dans combien de tissus le gène {gene} a-t-il été détecté par MS/MS ?"
        answer = f"Le gène {gene} a été détecté dans l'ensemble des 14 tissus analysés, indiquant une expression ubiquitaire."
    elif any(x in joined for x in ["none", "not_detected", "absent", "unknown"]):
        query = f"Le gène {gene} a-t-il été détecté par MS/MS dans un tissu ?"
        answer = f"Aucune détection du gène {gene} par MS/MS n’a été rapportée."
    elif n_tissues == 0:
        query = f"Le gène {gene} a-t-il été détecté par MS/MS dans un tissu ?"
        answer = f"Aucune détection du gène {gene} par MS/MS n’a été rapportée."
    elif n_tissues == 1:
        query = f"Dans combien de tissus le gène {gene} a-t-il été détecté par MS/MS ?"
        answer = f"Le gène {gene} a été détecté dans un seul tissu : {tissues[0]}."
    else:
        query = f"Dans combien de tissus le gène {gene} a-t-il été détecté par MS/MS ?"
        answer = f"Le gène {gene} a été détecté dans {n_tissues} tissus : {', '.join(tissues)}."
    return query, answer


def template_gene_to_tissues(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    """Génère des questions modulaires sur la distribution tissulaire du gène."""
    df_valid = df.drop_nulls(subset=["Symbol", "Tissues"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        gene = row["Symbol"]
        tissues = [t.strip() for t in str(row["Tissues"]).split(",") if t.strip()]
        chunk_ids = match_chunks_by_term(chunks, gene, source="mitocarta", category="gene")

        query, ground_truth = format_tissue_answer(gene, tissues)

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": query,
            "ground_truth": ground_truth,
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_tissues"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions



# ===============================
# 6️⃣ SUBMITOLOCALIZATION → GENES
# ===============================

def template_localization_to_genes(df: pl.DataFrame, chunks: list, n: int = 5) -> list[dict]:
    df_valid = df.drop_nulls(subset=["Symbol", "MitoCarta3.0_SubMitoLocalization"])
    locs = df_valid["MitoCarta3.0_SubMitoLocalization"].unique().to_list()
    loc_terms = set()
    for loc in locs:
        if isinstance(loc, str):
            loc_terms.update([x.strip() for x in loc.split("|") if x.strip()])
    sample_locs = random.sample(list(loc_terms), min(n, len(loc_terms)))

    questions = []
    for loc in sample_locs:
        genes = (
            df.filter(pl.col("MitoCarta3.0_SubMitoLocalization").str.contains(loc))
            .select("Symbol")
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        chunk_ids = match_chunks_by_term(chunks, loc, source="mitocarta", category="text")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quels gènes sont localisés dans la {loc} selon MitoCarta3.0 ?",
            "ground_truth": f"Les gènes localisés dans la {loc} sont : {', '.join(genes)}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["localization_to_genes"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# 7️⃣ PATHWAY → GENES
# ===============================

def template_pathway_to_genes(df: pl.DataFrame, chunks: list, n: int = 5) -> list[dict]:
    df_valid = df.drop_nulls(subset=["Symbol", "MitoCarta3.0_MitoPathways"])
    all_paths = df_valid["MitoCarta3.0_MitoPathways"].unique().to_list()
    paths = set()
    for p in all_paths:
        if isinstance(p, str):
            paths.update(re.split(r"[>|,|;]", p))
    sample_paths = random.sample(list(paths), min(n, len(paths)))

    questions = []
    for path in sample_paths:
        genes = (
            df.filter(pl.col("MitoCarta3.0_MitoPathways").str.contains(path.strip()))
            .select("Symbol")
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        chunk_ids = match_chunks_by_term(chunks, path, source="mitocarta", category="text")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quels gènes appartiennent à la voie {path.strip()} ?",
            "ground_truth": f"Les gènes associés à la voie {path.strip()} sont : {', '.join(genes)}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["pathway_to_genes"],
            "chunk_metadata": {"chunk_ids": chunk_ids, "source": SOURCE_DATASET}
        })
    return questions


# ===============================
# GÉNÉRATION TOTALE
# ===============================

def generate_all_mitocarta_questions(df, chunks, n_per_template=10):
    all_q = []
    all_q += template_ensembl_to_gene(df, chunks, n_per_template)
    all_q += template_gene_to_localization(df, chunks, n_per_template)
    all_q += template_gene_to_evidence(df, chunks, n_per_template)
    all_q += template_gene_to_proteinlength(df, chunks, n_per_template)
    all_q += template_gene_to_tissues(df, chunks, n_per_template)
    all_q += template_localization_to_genes(df, chunks, n_per_template)
    all_q += template_pathway_to_genes(df, chunks, n_per_template)
    return all_q


# ===============================
# MAIN
# ===============================

def main():
    print("📥 Chargement MitoCarta3.0 et chunks...")
    df = load_mitocarta(MITOCARTA_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    print("🧠 Génération multi-templates MitoCarta...")
    new_questions = generate_all_mitocarta_questions(df, chunks, n_per_template=10)

    print("💾 Sauvegarde...")
    existing = read_json_array(OUTPUT_PATH)
    write_json_array(OUTPUT_PATH, existing + new_questions)

    print(f"✅ {len(new_questions)} questions ajoutées.")
    print(f"📊 Total : {len(existing) + len(new_questions)} questions.")


if __name__ == "__main__":
    main()
