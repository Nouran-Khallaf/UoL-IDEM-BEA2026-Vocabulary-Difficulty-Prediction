from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import load_and_resolve_config, save_resolved_config
from src.core.exceptions import ConfigError, ExperimentRuntimeError
from src.data.load_raw import load_raw_dataset


SUPPORTED_SPLITS = {"train", "dev", "test", "all"}



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load raw BEA KVL data and materialize split-level feature-ready tables.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Which split to materialize: train, dev, test, or all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the feature-build output directory.",
    )
    parser.add_argument(
        "--save-resolved-config",
        action="store_true",
        help="Save resolved_config.yaml into the output directory.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a JSON summary after feature materialization.",
    )
    return parser.parse_args()



def _resolve_split(split_name: str) -> str:
    split_name = str(split_name).strip().lower()
    if split_name not in SUPPORTED_SPLITS:
        raise ConfigError(
            f"Unsupported split '{split_name}'. Allowed: {sorted(SUPPORTED_SPLITS)}"
        )
    return split_name



def _resolve_output_dir(cfg: dict[str, Any], cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        return cli_output_dir.resolve()

    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    experiment_name = cfg.get("experiment_name") or "feature_build"

    processed_dir = paths.get("processed_data_dir") or "data/processed"
    return (Path(processed_dir) / str(experiment_name)).resolve()



def _select_splits(loaded: dict[str, dict[str, Any]], split_name: str) -> dict[str, dict[str, Any]]:
    if split_name == "all":
        return loaded

    if split_name not in loaded:
        raise ExperimentRuntimeError(
            f"Requested split '{split_name}' was not loaded. Available: {sorted(loaded.keys())}"
        )
    return {split_name: loaded[split_name]}



def _write_split_outputs(output_dir: Path, split_name: str, df: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / f"{split_name}_features.csv", index=False)

    with (output_dir / f"{split_name}_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)



def _build_summary(
    *,
    experiment_name: str | None,
    output_dir: Path,
    selected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split_name, payload in selected.items():
        df = payload["df"]
        diagnostics = payload["diagnostics"]
        split_summaries[split_name] = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "file_path": diagnostics.get("file_path"),
            "l1_values": diagnostics.get("l1_values"),
            "target_nan_count": diagnostics.get("target_nan_count"),
        }

    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "output_dir": str(output_dir),
        "splits": split_summaries,
    }



def main() -> None:
    args = _parse_args()
    split_name = _resolve_split(args.split)

    resolved_config = load_and_resolve_config(args.config)
    output_dir = _resolve_output_dir(resolved_config, args.output_dir)

    loaded = load_raw_dataset(resolved_config)
    selected = _select_splits(loaded, split_name)

    for current_split, payload in selected.items():
        _write_split_outputs(
            output_dir=output_dir,
            split_name=current_split,
            df=payload["df"],
            diagnostics=payload["diagnostics"],
        )

    if args.save_resolved_config:
        save_resolved_config(resolved_config, output_dir / "resolved_config.yaml")

    summary = _build_summary(
        experiment_name=resolved_config.get("experiment_name"),
        output_dir=output_dir,
        selected=selected,
    )

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Feature build completed: {output_dir}")
        for current_split, payload in selected.items():
            df = payload["df"]
            print(f"- {current_split}: rows={len(df)}, cols={df.shape[1]}")


if __name__ == "__main__":
    main()
