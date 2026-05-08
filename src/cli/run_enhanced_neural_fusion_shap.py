from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.core.config import load_and_resolve_config, save_resolved_config
from src.core.exceptions import ExperimentRuntimeError
from src.pipelines.run_enhanced_neural_fusion import run_neural_fusion_experiment


SUPPORTED_EXPERIMENT_TYPES = {
    "neural_fusion",
    "enhanced_neural_fusion",
    "fusion_neural",
    "transformer_tabular_fusion",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an enhanced neural fusion BEA KVL experiment from a resolved/inheriting YAML config. "
            "This runner is for the transformer + engineered-feature late-fusion architecture "
            "with the enhanced fusion regressor."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the run output directory.",
    )
    parser.add_argument(
        "--save-resolved-config",
        action="store_true",
        help="Save resolved_config.yaml into the run output directory.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print compact JSON summary after the run finishes.",
    )

    # Explainability / SHAP options
    parser.add_argument(
        "--run-shap",
        action="store_true",
        help="Run SHAP expected-gradients for the tabular branch after training.",
    )
    parser.add_argument(
        "--shap-explain-max-rows",
        type=int,
        default=None,
        help="Number of dev rows to explain. Use None for full dev set.",
    )
    parser.add_argument(
        "--shap-background-max-rows",
        type=int,
        default=64,
        help="Number of train rows used as SHAP background/reference rows.",
    )
    parser.add_argument(
        "--shap-max-display",
        type=int,
        default=20,
        help="Maximum number of features shown in SHAP summary plots.",
    )
    parser.add_argument(
        "--shap-dependence-top-k",
        type=int,
        default=8,
        help="Number of top features for which to save colored SHAP dependence plots.",
    )

    return parser.parse_args()


def _override_output_dir(cfg: dict[str, Any], output_dir: Path | None) -> dict[str, Any]:
    if output_dir is None:
        return cfg

    cfg = dict(cfg)
    outputs = dict(cfg.get("outputs", {}))
    outputs["output_subdir"] = str(output_dir)
    cfg["outputs"] = outputs
    return cfg


def _inject_explainability_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    explainability = dict(cfg.get("explainability", {}))

    explainability["run_shap"] = bool(args.run_shap)
    explainability["shap_explain_max_rows"] = args.shap_explain_max_rows
    explainability["shap_background_max_rows"] = args.shap_background_max_rows
    explainability["shap_max_display"] = args.shap_max_display
    explainability["shap_dependence_top_k"] = args.shap_dependence_top_k

    cfg["explainability"] = explainability
    return cfg


def _resolve_experiment_type(cfg: dict[str, Any]) -> str:
    exp_type = cfg.get("experiment_type", "enhanced_neural_fusion")
    if not isinstance(exp_type, str) or not exp_type.strip():
        raise ExperimentRuntimeError("experiment_type must be a non-empty string.")

    exp_type = exp_type.strip().lower()
    if exp_type not in SUPPORTED_EXPERIMENT_TYPES:
        raise ExperimentRuntimeError(
            f"Unsupported experiment_type '{exp_type}' for this CLI. "
            f"Currently supported: {sorted(SUPPORTED_EXPERIMENT_TYPES)}"
        )
    return exp_type


def _print_summary_json(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = _parse_args()

    resolved_config = load_and_resolve_config(args.config)
    resolved_config = _override_output_dir(resolved_config, args.output_dir)
    resolved_config = _inject_explainability_config(resolved_config, args)

    experiment_type = _resolve_experiment_type(resolved_config)
    if experiment_type not in SUPPORTED_EXPERIMENT_TYPES:
        raise ExperimentRuntimeError(
            f"run_enhanced_neural_fusion_experiment.py expected one of "
            f"{sorted(SUPPORTED_EXPERIMENT_TYPES)}, got '{experiment_type}'."
        )

    result = run_neural_fusion_experiment(
        resolved_config=resolved_config,
    )

    if args.save_resolved_config:
        save_resolved_config(
            result.resolved_config,
            result.output_dir / "resolved_config.yaml",
        )

    summary = {
        "experiment_name": result.resolved_config.get("experiment_name"),
        "model_name": result.resolved_config.get("model_name", "enhanced_neural_fusion"),
        "text_columns": result.text_columns,
        "n_numeric_features": len(result.numeric_features),
        "numeric_features": result.numeric_features,
        "output_dir": str(result.output_dir),
        "oof_summary": result.oof_artifacts.summary,
        "dev_metrics": result.dev_metrics,
        "resolved_metadata": result.resolved_config.get("resolved_metadata", {}),
        "explainability": result.resolved_config.get("explainability", {}),
    }

    if args.print_summary:
        _print_summary_json(summary)
    else:
        print(f"Run completed: {result.output_dir}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()