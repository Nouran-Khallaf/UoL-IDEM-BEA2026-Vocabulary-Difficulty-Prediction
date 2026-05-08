#!/bin/bash
#SBATCH --job-name=cn_feeatures_shap
#SBATCH --output=logs/cn_features_shap-%j.out
#SBATCH --time=25:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=smlnkh@leeds.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

source miniconda3/etc/profile.d/conda.sh
conda activate mlenv

export WANDB_API_KEY=''
export HF_HOME="huggingface_cache"
export HUGGINGFACE_HUB_CACHE="huggingface_cache/hub"
export TRANSFORMERS_CACHE="huggingface_cache/hub"

mkdir -p logs
mkdir -p huggingface_cache/hub


#python -m src.cli.build_features_cli \
 # --config configs/experiments/de_all_features.yaml \
  #--split test \
  #--save-resolved-config \
  #--print-summary
#python -m src.cli.run_text_feature_regression --config configs/experiments/es_bge_m3_text_feature_regression_cognet_semantics_structural.yaml
#python -m src.cli.run_neural_fusion_cli --config configs/experiments/cn_multi_abilation.yaml --save-resolved-config --print-summary
python -m src.cli.run_enhanced_neural_fusion_shap \
   --save-resolved-config \
   --config configs/experiments/de_neurl_em_advanced.yaml \
   --print-summary \
   --run_shap  True\

#python -m src.cli.run_feature_target_correlation \
 # --feature-dir data/processed/cn_all_features \
#  --output-dir runs/cn_feature_correlation_new \
#  --encode-kelly-cefr \
##  --top-n 20 \
 # --score-column abs_kendall_tau
#python -m src.cli.extract_bge_embeddings --config configs/embeddings/cn_bge_extract.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_embedings_bge_ridge.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_gbr.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_xgb.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_svr.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_average_ensemble.yaml

#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_tabular_ridge.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_tabular_gbr.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_tabular_xgb.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/de_saved_bge_tabular_svr.yaml
#python -m src.cli.run_saved_embedding_regression --config configs/experiments/cn_saved_bge_tabular_average_ensemble.yaml

#python -m src.cli.run_ml_ensemble_experiment --feature-dir data/processed/es_all_features --output-dir runs/es_ml_ensemble_all --target-col GLMM_score --id-col item_id --n-splits 5 --seed 42 --use-xgb

#python -m src.cli.run_ml_ensemble_experiment \
 # --feature-dir data/processed/es_all_features\
  #--output-dir runs/es_ml_ensemble_all \
  #--target-col GLMM_score \
  #--id-col id \
  #--n-splits 5 \
  #--seed 42 \
  #--use-xgb

