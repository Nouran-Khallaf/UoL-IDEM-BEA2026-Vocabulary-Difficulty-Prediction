from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import load_and_resolve_config, save_resolved_config
from src.core.exceptions import ConfigError, ExperimentRuntimeError
from src.core.late_fusion_config import (
    apply_late_fusion_tabular_overrides,
    resolve_late_fusion_config,
)
from src.pipelines.run_tabular import run_tabular_experiment


SUPPORTED_EXPERIMENT_TYPES = {
    "tabular",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BEA KVL experiment from a resolved/inheriting YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Optional override for the training CSV/TSV file.",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=None,
        help="Optional override for the dev CSV/TSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the run output directory.",
    )
    parser.add_argument(
        "--numeric-features",
        nargs="*",
        default=None,
        help="Optional explicit numeric feature column names.",
    )
    parser.add_argument(
        "--categorical-features",
        nargs="*",
        default=None,
        help="Optional explicit categorical feature column names.",
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
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ConfigError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Dataset path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ConfigError(
        f"Unsupported dataset file extension '{suffix}' for {path}. "
        f"Supported: .csv, .tsv, .txt"
    )


def _resolve_experiment_type(cfg: dict[str, Any]) -> str:
    exp_type = cfg.get("experiment_type", "tabular")
    if not isinstance(exp_type, str) or not exp_type.strip():
        raise ExperimentRuntimeError("experiment_type must be a non-empty string.")
    exp_type = exp_type.strip().lower()

    if exp_type not in SUPPORTED_EXPERIMENT_TYPES:
        raise ExperimentRuntimeError(
            f"Unsupported experiment_type '{exp_type}' for this CLI. "
            f"Currently supported: {sorted(SUPPORTED_EXPERIMENT_TYPES)}"
        )
    return exp_type


def _resolve_train_path(cfg: dict[str, Any], cli_train_file: Path | None) -> Path:
    if cli_train_file is not None:
        return cli_train_file.resolve()

    files = cfg.get("files") if isinstance(cfg.get("files"), dict) else {}
    train_path = files.get("train")
    if not isinstance(train_path, str) or not train_path.strip():
        raise ExperimentRuntimeError(
            "Could not resolve training file from config. Expected files.train or --train-file."
        )
    return Path(train_path).resolve()


def _resolve_dev_path(cfg: dict[str, Any], cli_dev_file: Path | None) -> Path | None:
    if cli_dev_file is not None:
        return cli_dev_file.resolve()

    files = cfg.get("files") if isinstance(cfg.get("files"), dict) else {}
    dev_path = files.get("dev")
    if dev_path is None:
        return None
    if not isinstance(dev_path, str) or not dev_path.strip():
        raise ExperimentRuntimeError("If files.dev is provided, it must be a non-empty string.")
    return Path(dev_path).resolve()


def _override_output_dir(cfg: dict[str, Any], output_dir: Path | None) -> dict[str, Any]:
    if output_dir is None:
        return cfg

    cfg = dict(cfg)
    outputs = dict(cfg.get("outputs", {}))
    outputs["output_subdir"] = str(output_dir)
    cfg["outputs"] = outputs
    return cfg


def _apply_optional_late_fusion(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    If fusion.enabled=true and mode=late_fusion, redirect tabular_input
    to the embedding-augmented feature files.
    """
    late_fusion_cfg = resolve_late_fusion_config(cfg)
    if not late_fusion_cfg.enabled:
        return cfg

    return apply_late_fusion_tabular_overrides(cfg, late_fusion_cfg)


def _print_summary_json(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = _parse_args()

    resolved_config = load_and_resolve_config(args.config)
    resolved_config = _override_output_dir(resolved_config, args.output_dir)
    resolved_config = _apply_optional_late_fusion(resolved_config)

    experiment_type = _resolve_experiment_type(resolved_config)
    if experiment_type != "tabular":
        raise ExperimentRuntimeError(
            f"run_experiment_cli.py currently supports only tabular experiments, got '{experiment_type}'."
        )

    train_path = _resolve_train_path(resolved_config, args.train_file)
    _ = _resolve_dev_path(resolved_config, args.dev_file)

    train_df = _read_table(train_path)
    if train_df.empty:
        raise ExperimentRuntimeError(f"Training dataframe is empty: {train_path}")

    result = run_tabular_experiment(
        resolved_config=resolved_config,
        numeric_features=args.numeric_features,
        categorical_features=args.categorical_features,
    )

    if args.save_resolved_config:
        save_resolved_config(
            result.resolved_config,
            result.output_dir / "resolved_config.yaml",
        )

    if args.print_summary:
        _print_summary_json(result.oof_artifacts.summary)
    else:
        print(f"Run completed: {result.output_dir}")
        print(
            json.dumps(
                {
                    "experiment_name": result.resolved_config.get("experiment_name"),
                    "model_name": result.resolved_config.get("model_name"),
                    "n_features": len(result.features_used),
                    "output_dir": str(result.output_dir),
                    "oof_summary": result.oof_artifacts.summary,
                    "resolved_metadata": result.resolved_config.get("resolved_metadata", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()