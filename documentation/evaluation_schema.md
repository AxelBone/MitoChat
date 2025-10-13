# 🧩 Spécification du schéma JSON – *RAG Evaluation Record*

Ce schéma JSON définit la structure standardisée utilisée pour décrire et évaluer des **exemples RAG (Retrieval-Augmented Generation)** dans le projet.
Il garantit la **traçabilité**, la **comparabilité** et la **reproductibilité** des évaluations entre plusieurs sources de données (ClinVar, MitoCarta, PDF, etc.).

---

## 🎯 Objectif général

Le fichier JSON décrit, pour chaque *question d’évaluation*, toutes les informations nécessaires pour analyser :

* la **qualité de la réponse** générée par le modèle RAG,
* la **pertinence des contextes** documentaires utilisés,
* la **fidélité au corpus**,
* et la **difficulté cognitive de la tâche** selon le profil utilisateur.

Chaque enregistrement correspond à **une question RAG unique**.

---

## 📘 Structure générale

### Vue hiérarchique simplifiée

```
{
  id,
  query,
  ground_truth,
  context_v1[],
  context_v2[],
  type_question,
  difficulty,
  profile,
  metadata{},
  generated_answer,
  evaluation_scores{},
  feedback,
  linked_questions[],
  language,
  version
}
```

---

## 🧱 Champs obligatoires

| Champ             | Type          | Description                                                                                    |
| ----------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| **id**            | string        | Identifiant unique (UUID ou hash) pour chaque exemple.                                         |
| **query**         | string        | Question posée au système RAG.                                                                 |
| **ground_truth**  | string        | Réponse correcte et justifiée par les contextes (`context_v1`).                                |
| **type_question** | string (enum) | Catégorie cognitive : “faits-directs”, “synthèse multi-documents”, “raisonnement causal”, etc. |
| **difficulty**    | string (enum) | Niveau de complexité de la question : “facile”, “intermédiaire”, “difficile”.                  |
| **profile**       | string (enum) | Profil d’utilisateur visé : “patient” ou “professionnel”.                                      |

---

## 🧠 Champs de contenu

### `ground_truth` et `ungrounded_answer`

| Champ                 | Description                                                                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ground_truth**      | La réponse correcte, **vérifiée** dans les documents. Elle sert de référence pour les métriques *faithfulness* et *answer relevance*.                     |
| **ungrounded_answer** | Une réponse **plausible mais non justifiée** par les contextes. Elle permet d’évaluer la tendance du modèle à **halluciner** ou à extrapoler hors corpus. Elle est notamment générée sans RAG. |

Ces deux champs peuvent coexister pour tester la capacité du modèle à s’appuyer réellement sur les sources plutôt que sur sa connaissance interne.

---

### `context_v1` (contexte idéal minimal)

Liste des passages strictement nécessaires pour répondre correctement. Ces passages sont obtenus par le modèle RAG lors de l'évaluation.
Chaque élément est un objet contenant des métadonnées de provenance :

```json
{
  "text": "ClinVar: SURF1 est associé au syndrome de Leigh.",
  "source": "ClinVar",
  "document": "clinvar_2025.json",
  "page": 1,
  "position": "entry #234",
  "type": "text"
}
```

#### Champs internes

| Champ        | Description                                               |
| ------------ | --------------------------------------------------------- |
| **text**     | Passage textuel extrait du corpus.                        |
| **source**   | Origine du passage (ClinVar, MitoCarta, PDF…).            |
| **document** | Nom du fichier ou identifiant documentaire.               |
| **page**     | Numéro de page (si document PDF).                         |
| **position** | Ligne, paragraphe ou section d’origine.                   |
| **type**     | Nature du contenu : “text”, “table”, “image”, “metadata”. |

---

### `context_v2` (contexte bruité)

`context_v2` reprend `context_v1` **en y ajoutant des passages redondants ou non pertinents**.
Objectif : évaluer la *robustesse* du modèle à des contextes élargis ou mal filtrés.

---

## ⚙️ Métadonnées (`metadata`)

Bloc optionnel contenant les informations d’origine et d’annotation.

| Champ              | Description                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| **created_at**     | Horodatage ISO 8601 de la création.                                                                    |
| **annotator**      | Nom ou identifiant de l’annotateur.                                                                    |
| **validated**      | Booléen indiquant si la question a été validée.                                                        |
| **source_files**   | Liste des fichiers documentaires utilisés.                                                             |
| **auto_generated** | Indique quels champs ont été produits automatiquement (`query`, `ground_truth`, `contexts`, `labels`). |
| **tags**           | Liste de mots-clés thématiques (“mitochondrie”, “variant”, “diagnostic”).                              |

---

## 📊 Résultats d’évaluation

| Champ                 | Description                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **generated_answer**  | Réponse réellement produite par le système RAG à la question.                                      |
| **evaluation_scores** | Scores RAGAS calculés (`faithfulness`, `answer_relevance`, `context_precision`, `context_recall`). |
| **feedback**          | Commentaire libre d’un évaluateur humain.                                                          |
| **linked_questions**  | Liste d’autres questions du même fil de conversation.                                              |
| **language**          | Langue de la question et de la réponse (par défaut “fr”).                                          |
| **version**           | Version du corpus (ex. *ClinVar v2025*).                                                           |

---

## 🧬 Relations entre les champs

| Relation                                 | Description                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `ground_truth` ↔ `context_v1`            | La réponse doit être **justifiable** uniquement par les contextes de `context_v1`.    |
| `ungrounded_answer` ↔ `context_v1`       | Sert à mesurer les **hallucinations** : une réponse plausible mais absente du corpus. |
| `context_v2`                             | Contient `context_v1` + du **bruit** pour tester la robustesse à la redondance.       |
| `generated_answer` ↔ `evaluation_scores` | Permet le calcul des métriques RAGAS.                                                 |

---

## 🪄 Exemple complet

```json
{
  "id": "uuid-001",
  "query": "Quels gènes sont associés au syndrome de Leigh ?",
  "ground_truth": "Les gènes SURF1, NDUFS4 et MT-ATP6 sont associés à la maladie de Leigh.",
  "ungrounded_answer": "Le syndrome de Leigh est causé par un déficit énergétique global.",
  "context_v1": [
    {
      "text": "ClinVar: SURF1 et NDUFS4 sont associés au syndrome de Leigh.",
      "source": "ClinVar",
      "document": "clinvar_2025.json",
      "page": null,
      "position": "entry #122",
      "type": "text"
    }
  ],
  "context_v2": [
    {
      "text": "ClinVar: SURF1 et NDUFS4 sont associés au syndrome de Leigh.",
      "source": "ClinVar",
      "document": "clinvar_2025.json",
      "page": null,
      "position": "entry #122",
      "type": "text"
    },
    {
      "text": "MitoCarta3: Le gène POLG est lié à plusieurs troubles mitochondriaux.",
      "source": "MitoCarta3",
      "document": "mitocarta3_genes.json",
      "page": null,
      "position": "line 1200",
      "type": "text"
    }
  ],
  "type_question": "synthèse multi-documents",
  "difficulty": "difficile",
  "profile": "professionnel",
  "metadata": {
    "created_at": "2025-10-13T17:45:00Z",
    "annotator": "Alice",
    "validated": true,
    "source_files": ["clinvar_2025.json", "mitocarta3_genes.json"],
    "auto_generated": {
      "query": false,
      "ground_truth": true,
      "contexts": true,
      "labels": false
    },
    "tags": ["mitochondrie", "syndrome de Leigh", "SURF1"]
  },
  "generated_answer": "Les gènes SURF1 et NDUFS4 sont associés au syndrome de Leigh.",
  "evaluation_scores": {
    "faithfulness": 0.96,
    "answer_relevance": 0.92,
    "context_precision": 0.87,
    "context_recall": 0.81
  },
  "feedback": "Bonne couverture du contexte, réponse partielle mais correcte.",
  "language": "fr",
  "version": "ClinVar v2025"
}
```

---

## 🧭 En résumé

| Niveau              | Objectif                                                            |
| ------------------- | ------------------------------------------------------------------- |
| **Core fields**     | Décrire la question, la vérité de référence et le contexte minimal. |
| **Extended fields** | Évaluer la robustesse et la fidélité du modèle.                     |
| **Metadata**        | Garantir la traçabilité et la reproductibilité des expériences.     |

