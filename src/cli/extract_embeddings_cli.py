from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings.extract_text_embeddings import save_embedding_augmented_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate embedding-augmented feature CSVs from processed feature files. "
            "This is intended for late-fusion experiments where transformer embeddings "
            "are appended as dense tabular features."
        )
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the input processed feature CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to the output embedding-augmented CSV.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        required=True,
        help=(
            "Hugging Face encoder name, e.g. "
            "'bert-base-multilingual-cased', "
            "'xlm-roberta-base', "
            "'sentence-transformers/LaBSE'."
        ),
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        required=True,
        help=(
            "One or more text columns to concatenate and encode, e.g. "
            "L1_context L1_source_word en_target_word"
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="emb",
        help="Prefix for generated embedding feature columns.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy for sentence representation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    parser.add_argument(
        "--separator-text",
        type=str,
        default=" [SEP] ",
        help="String used to join multiple text columns before encoding.",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Disable tokenizer special tokens during encoding.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    save_embedding_augmented_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        encoder_name=args.encoder_name,
        text_columns=args.text_columns,
        prefix=args.prefix,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        device=args.device,
        add_special_tokens=not args.no_special_tokens,
        separator_text=args.separator_text,
    )

    print(f"Saved embedding-augmented CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()

"""python -m src.cli.extract_embeddings_cli \
  --input-csv data/processed/de_features_v1/train_features.csv \
  --output-csv data/processed/de_features_v1/train_features_labse.csv \
  --encoder-name sentence-transformers/LaBSE \
  --text-columns L1_context L1_source_word en_target_word \
  --prefix labse \
  --pooling cls \
  --batch-size 16 \
  --max-length 128"""
"""python -m src.cli.extract_embeddings_cli \
  --input-csv data/processed/de_features_v1/dev_features.csv \
  --output-csv data/processed/de_features_v1/dev_features_labse.csv \
  --encoder-name sentence-transformers/LaBSE \
  --text-columns L1_context L1_source_word en_target_word \
  --prefix labse \
  --pooling cls \
  --batch-size 16 \
  --max-length 128"""