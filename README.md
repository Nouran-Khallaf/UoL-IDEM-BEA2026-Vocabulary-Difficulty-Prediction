# UoL@IDEM BEA 2026 Vocabulary Difficulty Prediction

Code for the UoL@IDEM closed-track submission to the BEA 2026 Shared Task on **L1-aware Vocabulary Difficulty Prediction for English Learners**.

The system predicts a continuous English vocabulary difficulty score from an L1-specific context, an L1 source word, the English target word, clue information, and engineered linguistic features. It combines multilingual contextual encoders with feature-based regression and neural fusion models.

## Overview

The repository contains a config-driven pipeline for:

1. loading the BEA shared-task train/dev/test files;
2. building engineered features for frequency, lexical form, retrieval, masked-language-model predictability, surprisal, semantic-domain shift, and cognate similarity;
3. training regression models, including tabular baselines, late-fusion models, and neural fusion models;
4. generating development predictions, test predictions, and diagnostic outputs;
5. running feature-importance and error-analysis scripts used in the paper.

## Repository structure

```text
.
├── configs/
│   ├── data/                 # Per-language data configs
│   ├── embeddings/           # Embedding extraction configs
│   ├── experiments/          # Main experiment configs
│   ├── features/             # Feature-group configs
│   └── global.yaml           # Shared defaults
├── data/
│   ├── raw/                  # BEA train/dev/test files; not committed in public release
│   ├── processed/            # Generated feature tables; not committed by default
│   ├── external/             # External resources; not committed by default
│   └── resources/            # Small derived resources only
├── runs/                     # Generated run outputs; not committed by default
├── src/
│   ├── cli/                  # Command-line entry points
│   ├── core/                 # Config loading, validation, and utilities
│   ├── data/                 # Data loading and enrichment
│   ├── embeddings/           # Text-embedding extraction
│   ├── evaluation/           # Metrics and reporting
│   ├── features/             # Engineered feature builders
│   ├── models/               # Regression and neural-fusion models
│   └── pipelines/            # End-to-end experiment pipelines
├── tests/                    # Unit tests; to be completed
└──  requirements.txt
 
```

## Main system components

### Input

Each instance contains:

- `L1`: learner/source language (`es`, `de`, or `cn`);
- `L1_context`: source-language sentence or context;
- `L1_source_word`: source-language cue word;
- `en_target_word`: English target word;
- `en_target_clue`: clue pattern such as first letter plus blanks;
- `en_target_pos`: target part of speech;
- `GLMM_score`: gold difficulty score for train/dev data.

### Feature groups

The pipeline builds several complementary feature groups:

- **Frequency features**: KELLY, wordfreq, and SUBTLEX-based frequency signals.
- **Lexical and clue features**: target/source length, clue length, syllables, overlap and POS indicators.
- **Retrieval features**: multilingual lexical retrieval signals and candidate-rank information.
- **MLM and surprisal features**: masked-language-model probability, rank, entropy, PLL and subword surprisal.
- **Semantic features**: USAS tag overlap, entropy and semantic-domain shift.
- **Cognate features**: character n-gram overlap, weighted edit similarity, multilingual cosine similarity and CogNet links.

### Models

The code supports:

- feature-only regression models such as Ridge, SVR, Gradient Boosting and XGBoost;
- late-fusion models using frozen multilingual sentence embeddings;
- neural fusion models combining transformer text representations with engineered tabular features.

The main submitted architecture is the neural fusion model, implemented in `src/pipelines/run_neural_fusion.py` and `src/models/neural_fusion_regressor.py`.

## Installation

Create and activate a Python environment, then install the project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development use:

```bash
pip install -e .
```

The pipeline uses Hugging Face models such as `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `bert-base-multilingual-cased`, `xlm-roberta-base`, and `sentence-transformers/LaBSE`. These are downloaded automatically by `transformers` / `sentence-transformers` when first used.

## Data setup

The public repository should not include the official shared-task data or large external resources unless their licences explicitly allow redistribution.

Expected raw data layout:

```text
data/raw/
├── es_train.csv
├── es_dev.csv
├── es_test.csv
├── de_train.csv
├── de_dev.csv
├── de_test.csv
├── cn_train.csv
├── cn_dev.csv
└── cn_test.csv
```

Expected external resources, if feature extraction is run from scratch:

```text
data/external/
├── SUBTLEX-US.xlsx
├── en_m3.xls
└── CogNet-v2.0.tsv
```

If the external resources are not redistributed, add a short note in `data/external/README.md` explaining where to obtain them and where to place them locally.

## Reproducing the pipeline

### 1. Build engineered features

Build all train/dev/test features for a language:

```bash
python -m src.cli.build_features_cli \
  --config configs/experiments/es_all_features.yaml \
  --split all \
  --save-resolved-config \
  --print-summary
```

Replace `es_all_features.yaml` with:

```text
configs/experiments/de_all_features.yaml
configs/experiments/cn_all_features.yaml
```

for German and Chinese respectively.

### 2. Train neural-fusion models

Spanish BGE-M3 neural fusion:

```bash
python -m src.cli.run_neural_fusion_cli \
  --config configs/experiments/es_bge_m3_neural_fusion.yaml \
  --save-resolved-config \
  --print-summary
```

German multilingual-E5 neural fusion:

```bash
python -m src.cli.run_neural_fusion_cli \
  --config configs/experiments/de_multilingual_e5_large_neural_fusion.yaml \
  --save-resolved-config \
  --print-summary
```

Chinese multilingual-E5 neural fusion:

```bash
python -m src.cli.run_neural_fusion_cli \
  --config configs/experiments/cn_multilingual_e5_large_neural_fusion.yaml \
  --save-resolved-config \
  --print-summary
```

Alternative configs for ablations, saved embeddings and feature-selection variants are in `configs/experiments/`.

### 3. Run feature-only or saved-embedding experiments

Feature-only / ML ensemble example:

```bash
python -m src.cli.run_ml_ensemble_experiment \
  --feature-dir data/processed/es_all_features \
  --output-dir runs/es_ml_ensemble_all \
  --target-col GLMM_score \
  --id-col item_id \
  --n-splits 5 \
  --seed 42 \
  --use-xgb
```

Saved-embedding regression example:

```bash
python -m src.cli.run_saved_embedding_regression \
  --config configs/experiments/de_saved_bge_tabular_average_ensemble.yaml
```

### 4. Predict using a saved model

```bash
python -m src.cli.predict_saved_neural_fusion \
  --run-dir runs/es_bge_m3_neural_fusion \
  --test-features data/processed/es_all_features/test_features.csv \
  --output-file runs/es_bge_m3_neural_fusion/test_predictions.csv
```

For enhanced neural-fusion checkpoints, use:

```bash
python -m src.cli.predict_saved_enhanced_neural_fusion \
  --run-dir runs/<run_name> \
  --test-features data/processed/<language>_final_features/test_features.csv \
  --output-file runs/<run_name>/test_predictions.csv
```

### 5. Run feature analysis

```bash
python -m src.cli.run_feature_target_correlation \
  --feature-dir data/processed/es_all_features \
  --output-dir runs/es_feature_correlation \
  --encode-kelly-cefr \
  --top-n 20 \
  --score-column abs_kendall_tau
```

## Development results

The following development-set results are from the saved run summaries in `runs/`.

| Language | Saved run | RMSE | Pearson | Spearman | Kendall tau |
|---|---:|---:|---:|---:|---:|
| German | `de_bge_m3_neural_fusion` | 1.117 | 0.814 | 0.826 | 0.629 |
| Spanish | `es_bge_m3_neural_fusion` | 1.111 | 0.832 | 0.840 | 0.651 |
| Chinese | `cn_BEG_neural_fusion_all` | 0.975 | 0.831 | 0.839 | 0.648 |

Official submitted test RMSEs reported in the paper:

| Language | Test RMSE |
|---|---:|
| Spanish | 1.132 |
| German | 1.037 |
| Chinese | 0.891 |

## GitHub release notes

Before making the repository public, remove generated artifacts, local paths, credentials and files whose licences do not allow redistribution. In particular:

- remove hard-coded credentials from shell scripts;
- remove `runs/*/final_model.pt` unless using Git LFS or a separate release asset;
- remove `data/processed_old/`, `__pycache__/`, `.DS_Store`, `__MACOSX/`, and duplicated `copy` files;
- do not commit full external resources such as CogNet or SUBTLEX unless redistribution is allowed;
- replace local paths such as `/mnt/scratch/...` with relative paths;
- add a `.gitignore` before committing.

## Citation

If you use this repository, please cite the BEA 2026 shared-task paper and the UoL@IDEM system paper.

```bibtex
@inproceedings{felice-skidmore-2026-bea-vdp,
  title     = {Findings of the {BEA} 2026 Shared Task on Vocabulary Difficulty Prediction for English Learners},
  author    = {Felice, Mariano and Skidmore, Lucy},
  booktitle = {Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)},
  year      = {2026},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics}
}
```

Add the system-paper BibTeX entry here once the final proceedings metadata is available.

## Licence

Add the project licence before release. Recommended options are MIT or Apache-2.0 for code, with a separate note that the BEA data and external lexical resources remain under their original licences.
