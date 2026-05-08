## Configuration files

To keep the repository clean, we do **not** include every exploratory YAML file. The public release keeps one main neural-fusion template:

```text
configs/experiments/main_neural_fusion.yaml
```

Use this file as the starting point for all main runs. For a new experiment, copy the template and change only the run-specific fields:

```bash
cp configs/experiments/main_neural_fusion.yaml configs/experiments/es_bge_m3_neural_fusion.yaml
```

Then edit these fields:

| Field | What to change | Example |
|---|---|---|
| `experiment_name` | Name of the run | `es_bge_m3_neural_fusion` |
| `fusion.encoder_name` | Hugging Face encoder | `BAAI/bge-m3` |
| `tabular_input.feature_dir` | Language-specific feature folder | `data/processed/es_all_features` |
| `outputs.output_subdir` | Output folder under `runs/` | `es_bge_m3_neural_fusion` |
| `selection.selected_features` | Optional reduced feature list | comment it out to use all features |

Recommended final settings:

| Language | Encoder | Feature directory | Output name |
|---|---|---|---|
| Spanish | `BAAI/bge-m3` | `data/processed/es_all_features` | `es_bge_m3_neural_fusion` |
| German | `intfloat/multilingual-e5-large` | `data/processed/de_all_features` | `de_multilingual_e5_large_neural_fusion` |
| Chinese | `BAAI/bge-m3` or the encoder used for the saved final run | `data/processed/cn_all_features` | `cn_bge_m3_neural_fusion` |

Run the model with:

```bash
python -m src.cli.run_neural_fusion_cli \
  --config configs/experiments/main_neural_fusion.yaml \
  --save-resolved-config \
  --print-summary
```

For an edited copy, replace the config path:

```bash
python -m src.cli.run_neural_fusion_cli \
  --config configs/experiments/de_multilingual_e5_large_neural_fusion.yaml \
  --save-resolved-config \
  --print-summary
```

The resolved config saved in each `runs/<run_name>/` folder records the exact settings used for that run. This is better than committing every exploratory YAML file.

### What not to commit

Do not commit old or exploratory config files such as:

```text
*copy.yaml
*_abilation.yaml
*_top10_*.yaml
*_saved_bge_*.yaml
*_ridge_late_fusion.yaml
*_xgb_late_fusion.yaml
```

Keep these only locally or move them to an untracked `configs/archive/` folder. The GitHub repository should contain the clean template and, if needed, only the final paper configs.
