from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def _read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def merge_feature_splits(
    train_features_path: str,
    dev_features_path: str,
    output_features_path: str,
    train_diagnostics_path: str | None = None,
    dev_diagnostics_path: str | None = None,
    output_diagnostics_path: str | None = None,
    id_column: str = "item_id",
    target_column: str = "GLMM_score",
    drop_duplicate_ids: bool = False,
) -> None:
    train_df = _read_csv(train_features_path)
    dev_df = _read_csv(dev_features_path)

    if train_df.empty:
        raise ValueError("Train features CSV is empty.")
    if dev_df.empty:
        raise ValueError("Dev features CSV is empty.")

    if id_column not in train_df.columns:
        raise ValueError(f"{id_column!r} missing from train features.")
    if id_column not in dev_df.columns:
        raise ValueError(f"{id_column!r} missing from dev features.")

    if target_column not in train_df.columns:
        raise ValueError(f"{target_column!r} missing from train features.")
    if target_column not in dev_df.columns:
        raise ValueError(f"{target_column!r} missing from dev features.")

    # Keep only common columns so the final file is consistent
    common_cols = [c for c in train_df.columns if c in dev_df.columns]
    if id_column not in common_cols or target_column not in common_cols:
        raise ValueError(
            f"Common columns must include both {id_column!r} and {target_column!r}."
        )

    train_only = [c for c in train_df.columns if c not in dev_df.columns]
    dev_only = [c for c in dev_df.columns if c not in train_df.columns]

    if train_only:
        print("Columns only in train and dropped from final merge:")
        for c in train_only:
            print(f"  - {c}")

    if dev_only:
        print("Columns only in dev and dropped from final merge:")
        for c in dev_only:
            print(f"  - {c}")

    train_df = train_df[common_cols].copy()
    dev_df = dev_df[common_cols].copy()

    merged_df = pd.concat([train_df, dev_df], axis=0, ignore_index=True)

    # Basic validation
    if merged_df[target_column].isna().any():
        n_missing = int(merged_df[target_column].isna().sum())
        raise ValueError(
            f"Merged features contain {n_missing} missing values in {target_column!r}."
        )

    duplicate_mask = merged_df[id_column].duplicated(keep="first")
    n_duplicates = int(duplicate_mask.sum())

    if n_duplicates > 0:
        dup_ids = merged_df.loc[duplicate_mask, id_column].tolist()[:20]
        msg = (
            f"Found {n_duplicates} duplicate {id_column!r} values after merge. "
            f"Examples: {dup_ids}"
        )
        if drop_duplicate_ids:
            print(msg)
            print("Dropping later duplicates and keeping first occurrence.")
            merged_df = merged_df.drop_duplicates(subset=[id_column], keep="first").copy()
        else:
            raise ValueError(msg)

    # Save merged CSV
    output_path = Path(output_features_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved merged features to: {output_path}")
    print(f"Merged shape: {merged_df.shape}")
    print(f"Number of columns: {len(merged_df.columns)}")

    # Merge diagnostics if provided
    if output_diagnostics_path:
        train_diag = _read_json(train_diagnostics_path) if train_diagnostics_path else {}
        dev_diag = _read_json(dev_diagnostics_path) if dev_diagnostics_path else {}

        merged_diag = {
            "merged_from": {
                "train_features_path": str(train_features_path),
                "dev_features_path": str(dev_features_path),
                "train_diagnostics_path": str(train_diagnostics_path) if train_diagnostics_path else None,
                "dev_diagnostics_path": str(dev_diagnostics_path) if dev_diagnostics_path else None,
            },
            "rows": {
                "train": int(len(train_df)),
                "dev": int(len(dev_df)),
                "merged": int(len(merged_df)),
            },
            "columns": {
                "common_columns": common_cols,
                "n_common_columns": int(len(common_cols)),
                "train_only_dropped": train_only,
                "dev_only_dropped": dev_only,
            },
            "schema": {
                "id_column": id_column,
                "target_column": target_column,
            },
            "feature_columns": [
                c for c in common_cols if c not in {id_column, target_column}
            ],
            "source_train_diagnostics": train_diag,
            "source_dev_diagnostics": dev_diag,
        }

        _write_json(merged_diag, output_diagnostics_path)
        print(f"Saved merged diagnostics to: {output_diagnostics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge already-generated train and dev feature CSV files."
    )
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--output-features", required=True)

    parser.add_argument("--train-diagnostics", default=None)
    parser.add_argument("--dev-diagnostics", default=None)
    parser.add_argument("--output-diagnostics", default=None)

    parser.add_argument("--id-column", default="item_id")
    parser.add_argument("--target-column", default="GLMM_score")
    parser.add_argument(
        "--drop-duplicate-ids",
        action="store_true",
        help="Drop later duplicate item_id rows instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_feature_splits(
        train_features_path=args.train_features,
        dev_features_path=args.dev_features,
        output_features_path=args.output_features,
        train_diagnostics_path=args.train_diagnostics,
        dev_diagnostics_path=args.dev_diagnostics,
        output_diagnostics_path=args.output_diagnostics,
        id_column=args.id_column,
        target_column=args.target_column,
        drop_duplicate_ids=args.drop_duplicate_ids,
    )


if __name__ == "__main__":
    main()