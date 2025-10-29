"""
clinvar_auto_generate_modular.py
--------------------------------
Génération modulaire de questions unitaires ClinVar, avec plusieurs templates.
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
# CONFIGURATION GÉNÉRALE
# ===============================

DATA_DIR = Path("data")
CLINVAR_PATH = DATA_DIR / "clinvar" / "clinvar_mito_records.json"
CHUNKS_PATH = DATA_DIR / "chunks_generated" /  "chunks.jsonl"
OUTPUT_PATH = DATA_DIR / "annotations" / "questions_annotations.json"

SOURCE_DATASET = "clinvar"
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


def normalize_text(text: str) -> str:
    """Simplifie le texte pour un matching tolérant."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def match_chunks_by_term(
    chunks: list,
    term,
    source: str = None,
    category: str = None
) -> list[int]:
    """
    Cherche des chunks correspondant à un ou plusieurs termes dans une source donnée.

    Args:
        chunks (list): Liste des chunks enrichis.
        term (str | list[str]): Mot-clé ou liste de mots-clés à chercher.
        source (str, optional): Filtrer par source ('clinvar', 'mitocarta', 'genereviews', ...).
        category (str, optional): Type de champ privilégié ('variant', 'gene', 'disease', 'text', ...).

    Returns:
        list[int]: Liste triée et unique d'idx de chunks correspondants.
    """

    if not term:
        return []

    # 🧠 Uniformiser : convertir en liste
    if isinstance(term, str):
        search_terms = [term]
    elif isinstance(term, list):
        search_terms = [t for t in term if isinstance(t, str) and t.strip()]
    else:
        return []

    # Normaliser les termes
    search_terms_norm = [normalize_text(t) for t in search_terms]
    matched = []

    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        ch_source = str(meta.get("source", "")).lower()

        # 🧭 Filtrage par source
        if source and source.lower() not in ch_source:
            continue

        # 🧩 Déterminer les champs pertinents selon la catégorie et la source
        haystack_parts = []

        # ===== ClinVar =====
        if ch_source == "clinvar":
            if category == "variant":
                haystack_parts = [
                    meta.get("variant_id", ""),
                    meta.get("clnhgvs", ""),
                    meta.get("rsid", ""),
                    ch.get("text", ""),
                ]
            elif category == "gene":
                haystack_parts = [
                    meta.get("gene_symbol", ""),
                    meta.get("gene_id", ""),
                    ch.get("text", ""),
                ]
            elif category == "disease":
                diseases = meta.get("diseases", [])
                if isinstance(diseases, list):
                    haystack_parts = diseases
                elif isinstance(diseases, str):
                    haystack_parts = [diseases]
                haystack_parts += [ch.get("text", "")]
            else:
                haystack_parts = [ch.get("text", "")]

        # ===== MitoCarta =====
        elif ch_source == "mitocarta":
            haystack_parts = [
                meta.get("gene_symbol", ""),
                meta.get("gene_name", ""),
                meta.get("function", ""),
                ch.get("text", ""),
            ]

        # ===== GeneReviews =====
        elif ch_source == "genereviews":
            haystack_parts = [
                meta.get("doc_title", ""),
                meta.get("section", ""),
                ch.get("text", ""),
            ]

        # Concaténer et normaliser
        haystack = " ".join(map(str, haystack_parts))
        norm_haystack = normalize_text(haystack)

        # 🔍 Vérifier si un des termes apparaît dans le texte normalisé
        if any(t in norm_haystack for t in search_terms_norm):
            matched.append(ch.get("idx"))

    # 🔁 Retirer doublons et trier
    return sorted(set(filter(None, matched)))


# ===============================
# CHARGEMENT DES DONNÉES
# ===============================

def expand_info_column(df: pl.DataFrame) -> pl.DataFrame:
    """Aplati la colonne 'info' (dictionnaire) pour extraire les sous-champs info.CLNDN, CLNSIG, etc."""
    if "info" not in df.columns:
        return df

    info_dicts = df["info"].to_list()
    if not info_dicts or not isinstance(info_dicts[0], dict):
        return df

    # Récupère toutes les clés présentes dans les dicts de info
    all_keys = set()
    for entry in info_dicts:
        if isinstance(entry, dict):
            all_keys.update(entry.keys())

    # Transforme chaque dict de 'info' en colonnes séparées
    info_expanded = pl.DataFrame([
        {key: (entry.get(key) if isinstance(entry, dict) else None) for key in all_keys}
        for entry in info_dicts
    ])

    # Préfixe les colonnes et concatène
    info_expanded = info_expanded.rename({k: f"info.{k}" for k in all_keys})
    df_no_info = df.drop("info")
    return df_no_info.hstack(info_expanded)

def load_clinvar(path: Path) -> pl.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    df = pl.DataFrame(data)
    df = expand_info_column(df)
    print("✅ Colonnes ClinVar chargées :", df.columns)
    return df

def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    

def extract_clean_diseases(df: pl.DataFrame, column: str = "info.CLNDN") -> list[dict]:
    """
    Extrait la liste des maladies à partir du champ ClinVar.
    - Conserve 'raw' (avec underscores) pour les recherches exactes dans le DataFrame.
    - Produit 'clean' (espaces à la place des underscores) pour l'affichage et les questions.
    - Gère les séparateurs multiples | , / et nettoie les doublons.
    Retourne : [{"raw": "...", "clean": "..."}]
    """
    if column not in df.columns:
        print(f"⚠️ Colonne {column} absente du DataFrame ClinVar")
        return []

    # Liste brute de toutes les valeurs CLNDN
    diseases_raw_entries = df.select(pl.col(column)).drop_nulls().to_series().to_list()

    all_diseases = {}

    for entry in diseases_raw_entries:
        if not isinstance(entry, str):
            continue

        # Multi-split sur | , /
        parts = re.split(r"[|,/,]", entry)

        for disease in parts:
            raw = disease.strip()
            if not raw:
                continue

            # Filtrer les valeurs inutiles
            if raw.lower() in {"not_provided", "not_specified", "see_cases"}:
                continue

            # ✅ Version 'clean' (affichage humain)
            clean = raw.replace("_", " ").replace("  ", " ")

            # On garde les deux versions
            all_diseases[raw] = clean

    # Liste unique et triée
    diseases_list = [{"raw": k, "clean": v} for k, v in sorted(all_diseases.items(), key=lambda x: x[1].lower())]
    return diseases_list




# ===============================
# 1️⃣ VARIANT -> GENE
# ===============================

def template_variant_to_gene(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    """Quel gène est lié au variant {variant_id}?"""
    df_valid = df.drop_nulls(subset=["variant_id", "gene_symbol", "gene_id"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        variant_id = row["variant_id"]
        gene_symbol = row["gene_symbol"]
        gene_id = row["gene_id"]
        chunk_ids = match_chunks_by_term(chunks, variant_id, source="clinvar", category="variant_id")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quel gène est lié au variant {variant_id} ?",
            "ground_truth": f"Le variant {variant_id} est lié au gène {gene_symbol} (ID gène : {gene_id}).",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["variant_to_gene"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET,
                "doc_title": None,
                "section": None,
                "page": None,
                "block_type": None
            }
        })
    return questions


# ===============================
# 2️⃣ VARIANT -> CLINICAL SIGNIFICANCE
# ===============================

def template_variant_to_significance(df: pl.DataFrame, chunks: list, n: int = 10) -> list[dict]:
    """Quelle est la signification clinique du variant {variant_id}?"""
    df_valid = df.drop_nulls(subset=["variant_id", "clinical_significance"])
    df_sample = df_valid.sample(min(n, df_valid.height))
    questions = []

    for row in df_sample.iter_rows(named=True):
        variant_id = row["variant_id"]
        clinical_sign = row["clinical_significance"]
        chunk_ids = match_chunks_by_term(chunks, variant_id, source="clinvar", category="variant_id")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quelle est la signification clinique du variant {variant_id} ?",
            "ground_truth": f"Le variant {variant_id} est classé comme {clinical_sign}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["variant_to_significance"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET,
                "doc_title": None,
                "section": None,
                "page": None,
                "block_type": None
            }
        })
    return questions

# ===============================
# 3️⃣ DISEASE → VARIANTS
# ===============================

def template_disease_to_variants(df: pl.DataFrame, chunks: list, n: int = 1):
    df_valid = df.drop_nulls(subset=["info.CLNDN", "variant_id"])
    disease_pairs = extract_clean_diseases(df_valid, "info.CLNDN")
    if not disease_pairs:
        return []

    sample_diseases = random.sample(disease_pairs, min(n, len(disease_pairs)))
    # print("sample_diseases", sample_diseases)

    questions = []
    for pair in sample_diseases:
        disease_raw = pair["raw"]      # for variant research
        disease_clean = pair["clean"]  # for question rendering

        variants = (
            df.filter(pl.col("info.CLNDN").str.contains(disease_raw, literal=True))
            .select("variant_id")
            .unique()
        )
        # print(variants)

        variant_list = [v for v in variants["variant_id"]]
        if not variant_list:
            continue

        # print("disease_clean", disease_clean)
        # print("sum chunks",sum(disease_clean in ch.get("text", "").lower() for ch in chunks))
        chunk_ids = match_chunks_by_term(chunks, disease_clean, source="clinvar", category = "diseases")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quels variants sont associés à la maladie {disease_clean} ?",
            "ground_truth": f"Les variants {', '.join(variant_list)} sont associés à {disease_clean}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["disease_to_variants"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET
            }
        })
    return questions


# ===============================
# 4️⃣ DISEASE → GENES
# ===============================

def template_disease_to_genes(df: pl.DataFrame, chunks: list, n: int = 10):
    df_valid = df.drop_nulls(subset=["info.CLNDN", "gene_symbol"])
    disease_pairs = extract_clean_diseases(df_valid, "info.CLNDN")
    if not disease_pairs:
        return []

    sample_diseases = random.sample(disease_pairs, min(n, len(disease_pairs)))

    questions = []
    for pair in sample_diseases:
        disease_raw = pair["raw"]      # pour recherche chunks
        disease_clean = pair["clean"]  # pour affichage question

        # ⚠️ Recherche littérale
        genes = (
            df.filter(pl.col("info.CLNDN").str.contains(disease_raw, literal=True))
            .select("gene_symbol")
            .unique()
        )

        gene_list = [g for g in genes["gene_symbol"]]
        if not gene_list:
            continue


        chunk_ids = match_chunks_by_term(chunks, disease_clean, source="clinvar", category="diseases")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quels gènes sont associés à la maladie {disease_clean} ?",
            "ground_truth": f"Les gènes {', '.join(gene_list)} sont associés à {disease_clean}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["disease_to_genes"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET
            }
        })
    return questions



# ===============================
# 5️⃣ GENE → DISEASES
# ===============================

def template_gene_to_diseases(df: pl.DataFrame, chunks: list, n: int = 10):
    df_valid = df.drop_nulls(subset=["gene_symbol", "info.CLNDN"])
    genes = df_valid["gene_symbol"].unique().to_list()
    sample_genes = random.sample(genes, min(n, len(genes)))
    questions = []

    for gene in sample_genes:
        diseases = (
            df.filter(pl.col("gene_symbol") == gene)
            .select("info.CLNDN")
            .unique()
        )
        dis_list = []
        for d in diseases["info.CLNDN"]:
            if isinstance(d, str):
                dis_list.extend([x.strip() for x in d.split("|") if x not in ["not_provided", "not_specified"]])
        if not dis_list:
            continue

        chunk_ids = match_chunks_by_term(chunks, gene, source="clinvar", category="gene")

        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quelles maladies sont associées au gène {gene} ?",
            "ground_truth": f"Le gène {gene} est associé à {', '.join(sorted(set(dis_list)))}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_diseases"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET
            }
        })
    return questions


# ===============================
# 6️⃣ GENE → VARIANTS
# ===============================

def template_gene_to_variants(df: pl.DataFrame, chunks: list, n: int = 10):
    df_valid = df.drop_nulls(subset=["gene_symbol", "variant_id"])
    genes = df_valid["gene_symbol"].unique().to_list()
    sample_genes = random.sample(genes, min(n, len(genes)))
    questions = []

    for gene in sample_genes:
        variants = (
            df.filter(pl.col("gene_symbol") == gene)
            .select("variant_id")
            .unique()
        )
        var_list = [v for v in variants["variant_id"]]
        if not var_list:
            continue
        chunk_ids = match_chunks_by_term(chunks, gene)
        questions.append({
            "question_id": random_hex(16),
            "created_at": utc_now_iso(),
            "query": f"Quels variants du gène {gene} sont connus ?",
            "ground_truth": f"Les variants connus du gène {gene} sont {', '.join(var_list)}.",
            "type_question": "faits-directs",
            "difficulty": DIFFICULTY,
            "profile": PROFILE,
            "status": STATUS,
            "source_dataset": SOURCE_DATASET,
            "tags": ["gene_to_variants"],
            "chunk_metadata": {
                "chunk_ids": chunk_ids,
                "source": SOURCE_DATASET
            }
        })
    return questions



# ===============================
# 3️⃣ MANAGER — Générer tous les templates
# ===============================

def generate_all_clinvar_questions(df, chunks, n_per_template=10):
    all_q = []
    all_q += template_variant_to_gene(df, chunks, n_per_template)
    all_q += template_variant_to_significance(df, chunks, n_per_template)
    all_q += template_disease_to_variants(df, chunks, n_per_template)
    all_q += template_disease_to_genes(df, chunks, n_per_template)
    all_q += template_gene_to_diseases(df, chunks, n_per_template)
    all_q += template_gene_to_variants(df, chunks, n_per_template)
    return all_q


# ===============================
# MAIN
# ===============================

def main():
    print("📥 Chargement ClinVar et chunks...")
    df = load_clinvar(CLINVAR_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    print("🧠 Génération multi-templates ClinVar...")
    new_questions = generate_all_clinvar_questions(df, chunks, n_per_template=15)

    print("💾 Sauvegarde...")
    existing = read_json_array(OUTPUT_PATH)
    write_json_array(OUTPUT_PATH, existing + new_questions)

    print(f"✅ {len(new_questions)} questions ajoutées au total.")
    print(f"📊 Total : {len(existing) + len(new_questions)} questions.")


if __name__ == "__main__":
    main()
