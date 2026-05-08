from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import load_and_resolve_config, save_resolved_config
from src.core.exceptions import ConfigError
from src.pipelines.build_features import build_features_pipeline


SUPPORTED_SPLITS = {"train", "dev", "test", "all"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build engineered feature tables for BEA KVL experiments.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--split", type=str, default="all", help="train, dev, test, or all")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--save-resolved-config", action="store_true", help="Save resolved config YAML.")
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def _resolve_split(split_name: str) -> list[str] | None:
    split_name = str(split_name).strip().lower()
    if split_name not in SUPPORTED_SPLITS:
        raise ConfigError(f"Unsupported split '{split_name}'. Allowed: {sorted(SUPPORTED_SPLITS)}")
    if split_name == "all":
        return None
    return [split_name]


def main() -> None:
    args = _parse_args()
    splits_to_build = _resolve_split(args.split)

    resolved_config = load_and_resolve_config(args.config)

    result = build_features_pipeline(
        resolved_config,
        splits_to_build=splits_to_build,
        save_outputs=True,
        output_dir=args.output_dir,
    )

    if args.save_resolved_config:
        save_resolved_config(resolved_config, result.output_dir / "resolved_config.yaml")

    summary = {
        "status": "ok",
        "experiment_name": resolved_config.get("experiment_name"),
        "output_dir": str(result.output_dir),
        "splits": {
            split_name: {
                "n_rows": int(split_result.df.shape[0]),
                "n_columns": int(split_result.df.shape[1]),
                "feature_groups_applied": split_result.feature_groups_applied,
                "feature_columns": split_result.feature_columns,
            }
            for split_name, split_result in result.splits.items()
        },
    }

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Feature build completed: {result.output_dir}")
        for split_name, split_result in result.splits.items():
            print(
                f"- {split_name}: rows={split_result.df.shape[0]}, "
                f"cols={split_result.df.shape[1]}, "
                f"feature_cols={len(split_result.feature_columns)}"
            )


if __name__ == "__main__":
    main()