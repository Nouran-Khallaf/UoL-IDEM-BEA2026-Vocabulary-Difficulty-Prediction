from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.update_feature_diagnostics import (
    clone_and_update_train_dev_diagnostics,
    update_feature_diagnostics_with_embeddings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update processed feature diagnostics JSON so newly added embedding "
            "columns are included in feature_columns."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--single",
        action="store_true",
        help="Update one diagnostics file.",
    )
    mode.add_argument(
        "--train-dev",
        action="store_true",
        help="Update both train and dev diagnostics files together.",
    )

    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)

    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--input-diagnostics", type=str, default=None)
    parser.add_argument("--output-diagnostics", type=str, default=None)

    parser.add_argument("--train-csv", type=str, default="train_features.csv")
    parser.add_argument("--dev-csv", type=str, default="dev_features.csv")
    parser.add_argument("--train-diagnostics", type=str, default="train_feature_diagnostics.json")
    parser.add_argument("--dev-diagnostics", type=str, default="dev_feature_diagnostics.json")
    parser.add_argument("--output-train-diagnostics", type=str, default="train_feature_diagnostics.json")
    parser.add_argument("--output-dev-diagnostics", type=str, default="dev_feature_diagnostics.json")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feature_dir = args.feature_dir.resolve()

    if args.single:
        if not args.input_csv or not args.input_diagnostics:
            raise ValueError(
                "--single mode requires --input-csv and --input-diagnostics."
            )

        result = update_feature_diagnostics_with_embeddings(
            input_csv=feature_dir / args.input_csv,
            input_diagnostics=feature_dir / args.input_diagnostics,
            output_diagnostics=(
                feature_dir / args.output_diagnostics
                if args.output_diagnostics is not None
                else None
            ),
            prefix=args.prefix,
        )

        print("Updated diagnostics:")
        print(f"  input_csv: {result.input_csv}")
        print(f"  input_diagnostics: {result.input_diagnostics}")
        print(f"  output_diagnostics: {result.output_diagnostics}")
        print(f"  original_feature_columns: {result.n_original_feature_columns}")
        print(f"  new_embedding_columns: {result.n_new_embedding_columns}")
        print(f"  final_feature_columns: {result.n_final_feature_columns}")

    else:
        results = clone_and_update_train_dev_diagnostics(
            feature_dir=feature_dir,
            train_csv=args.train_csv,
            dev_csv=args.dev_csv,
            train_diagnostics=args.train_diagnostics,
            dev_diagnostics=args.dev_diagnostics,
            output_train_diagnostics=args.output_train_diagnostics,
            output_dev_diagnostics=args.output_dev_diagnostics,
            prefix=args.prefix,
        )

        for split_name, result in results.items():
            print(f"[{split_name}]")
            print(f"  input_csv: {result.input_csv}")
            print(f"  input_diagnostics: {result.input_diagnostics}")
            print(f"  output_diagnostics: {result.output_diagnostics}")
            print(f"  original_feature_columns: {result.n_original_feature_columns}")
            print(f"  new_embedding_columns: {result.n_new_embedding_columns}")
            print(f"  final_feature_columns: {result.n_final_feature_columns}")


if __name__ == "__main__":
    main()

"""python -m src.cli.update_feature_diagnostics_cli \
  --train-dev \
  --feature-dir data/processed/de_features_v1 \
  --train-csv train_features_labse.csv \
  --dev-csv dev_features_labse.csv \
  --train-diagnostics train_feature_diagnostics.json \
  --dev-diagnostics dev_feature_diagnostics.json \
  --output-train-diagnostics train_feature_diagnostics_labse.json \
  --output-dev-diagnostics dev_feature_diagnostics_labse.json \
  --prefix labse"""