from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.encoder_registry import resolve_encoder_params
from src.embeddings.extract_text_embeddings import save_embedding_augmented_csv
from src.embeddings.update_feature_diagnostics import (
    update_feature_diagnostics_with_embeddings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build embedding-augmented train/dev feature files and matching "
            "diagnostics files for late-fusion experiments."
        )
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory containing processed train/dev feature CSVs and diagnostics JSON files.",
    )

    parser.add_argument(
        "--train-file",
        type=str,
        default="train_features.csv",
        help="Base processed training feature CSV filename.",
    )
    parser.add_argument(
        "--dev-file",
        type=str,
        default="dev_features.csv",
        help="Base processed dev feature CSV filename.",
    )
    parser.add_argument(
        "--train-diagnostics-file",
        type=str,
        default="train_feature_diagnostics.json",
        help="Base training diagnostics JSON filename.",
    )
    parser.add_argument(
        "--dev-diagnostics-file",
        type=str,
        default="dev_feature_diagnostics.json",
        help="Base dev diagnostics JSON filename.",
    )

    encoder_group = parser.add_mutually_exclusive_group(required=True)
    encoder_group.add_argument(
        "--encoder-key",
        type=str,
        default=None,
        help="Registry key for a supported encoder, e.g. mbert, xlmr_base, labse.",
    )
    encoder_group.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="Explicit Hugging Face encoder name.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional override for embedding column prefix.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean"],
        default=None,
        help="Optional override for pooling strategy.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for maximum tokenized sequence length.",
    )

    parser.add_argument(
        "--text-columns",
        nargs="+",
        required=True,
        help=(
            "Text columns to concatenate and encode, e.g. "
            "L1_context L1_source_word en_target_word"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--separator-text",
        type=str,
        default=" [SEP] ",
        help="Separator string used between text fields before encoding.",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Disable tokenizer special tokens during encoding.",
    )

    parser.add_argument(
        "--train-output-file",
        type=str,
        default=None,
        help="Optional explicit output filename for train augmented CSV.",
    )
    parser.add_argument(
        "--dev-output-file",
        type=str,
        default=None,
        help="Optional explicit output filename for dev augmented CSV.",
    )
    parser.add_argument(
        "--train-output-diagnostics",
        type=str,
        default=None,
        help="Optional explicit output filename for train diagnostics JSON.",
    )
    parser.add_argument(
        "--dev-output-diagnostics",
        type=str,
        default=None,
        help="Optional explicit output filename for dev diagnostics JSON.",
    )

    return parser.parse_args()


def _default_augmented_name(base_filename: str, suffix: str) -> str:
    path = Path(base_filename)
    return f"{path.stem}_{suffix}{path.suffix}"


def _default_augmented_diag_name(base_filename: str, suffix: str) -> str:
    path = Path(base_filename)
    return f"{path.stem}_{suffix}{path.suffix}"


def main() -> None:
    args = _parse_args()
    feature_dir = args.feature_dir.resolve()

    encoder_params = resolve_encoder_params(
        encoder_key=args.encoder_key,
        encoder_name=args.encoder_name,
        prefix=args.prefix,
        pooling=args.pooling,
        max_length=args.max_length,
    )

    prefix = str(encoder_params["prefix"])
    encoder_name = str(encoder_params["encoder_name"])
    pooling = str(encoder_params["pooling"])
    max_length = int(encoder_params["max_length"])

    train_output_file = (
        args.train_output_file
        if args.train_output_file is not None
        else _default_augmented_name(args.train_file, prefix)
    )
    dev_output_file = (
        args.dev_output_file
        if args.dev_output_file is not None
        else _default_augmented_name(args.dev_file, prefix)
    )

    train_output_diagnostics = (
        args.train_output_diagnostics
        if args.train_output_diagnostics is not None
        else _default_augmented_diag_name(args.train_diagnostics_file, prefix)
    )
    dev_output_diagnostics = (
        args.dev_output_diagnostics
        if args.dev_output_diagnostics is not None
        else _default_augmented_diag_name(args.dev_diagnostics_file, prefix)
    )

    train_input_csv = feature_dir / args.train_file
    dev_input_csv = feature_dir / args.dev_file
    train_input_diag = feature_dir / args.train_diagnostics_file
    dev_input_diag = feature_dir / args.dev_diagnostics_file

    train_output_csv = feature_dir / train_output_file
    dev_output_csv = feature_dir / dev_output_file
    train_output_diag = feature_dir / train_output_diagnostics
    dev_output_diag = feature_dir / dev_output_diagnostics

    save_embedding_augmented_csv(
        input_csv=train_input_csv,
        output_csv=train_output_csv,
        encoder_name=encoder_name,
        text_columns=args.text_columns,
        prefix=prefix,
        batch_size=args.batch_size,
        max_length=max_length,
        pooling=pooling,
        device=args.device,
        add_special_tokens=not args.no_special_tokens,
        separator_text=args.separator_text,
    )

    update_feature_diagnostics_with_embeddings(
        input_csv=train_output_csv,
        input_diagnostics=train_input_diag,
        output_diagnostics=train_output_diag,
        prefix=prefix,
    )

    if dev_input_csv.exists() and dev_input_diag.exists():
        save_embedding_augmented_csv(
            input_csv=dev_input_csv,
            output_csv=dev_output_csv,
            encoder_name=encoder_name,
            text_columns=args.text_columns,
            prefix=prefix,
            batch_size=args.batch_size,
            max_length=max_length,
            pooling=pooling,
            device=args.device,
            add_special_tokens=not args.no_special_tokens,
            separator_text=args.separator_text,
        )

        update_feature_diagnostics_with_embeddings(
            input_csv=dev_output_csv,
            input_diagnostics=dev_input_diag,
            output_diagnostics=dev_output_diag,
            prefix=prefix,
        )

    print("Built embedding feature set:")
    print(f"  encoder_name: {encoder_name}")
    print(f"  prefix: {prefix}")
    print(f"  pooling: {pooling}")
    print(f"  max_length: {max_length}")
    print(f"  train_output_csv: {train_output_csv}")
    print(f"  train_output_diagnostics: {train_output_diag}")

    if dev_input_csv.exists() and dev_input_diag.exists():
        print(f"  dev_output_csv: {dev_output_csv}")
        print(f"  dev_output_diagnostics: {dev_output_diag}")


if __name__ == "__main__":
    main()

"""python -m src.cli.build_embedding_feature_set \
  --feature-dir data/processed/de_features_v1 \
  --encoder-key labse \
  --text-columns L1_context L1_source_word en_target_word \
  --batch-size 16 \
  --max-length 128"""

"""python -m src.cli.build_embedding_feature_set \
  --feature-dir data/processed/de_features_v1 \
  --encoder-key xlmr_base \
  --text-columns L1_context L1_source_word en_target_word \
  --batch-size 16 \
  --max-length 128"""
"""python -m src.cli.build_embedding_feature_set \
  --feature-dir data/processed/es_features_v1 \
  --encoder-key mbert \
  --text-columns L1_context L1_source_word en_target_word \
  --batch-size 16"""