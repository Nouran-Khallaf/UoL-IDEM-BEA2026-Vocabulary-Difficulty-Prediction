from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
import csv
import pandas as pd


# =========================================================
# BEA language setup
"""reader = pd.read_csv(
    cognet_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    engine="python",
    quoting=csv.QUOTE_NONE,
    on_bad_lines=_handle_bad_line,
    chunksize=chunksize,
    usecols=range(7),
    encoding="utf-8",
)"""
# =========================================================
BEA_SOURCE_LANGS = {"spa", "deu", "cmn", "zho"}
TARGET_LANG = "eng"

EXPECTED_RAW_COLUMNS = [
    "concept id",
    "lang 1",
    "word 1",
    "lang 2",
    "word 2",
    "translit 1",
    "translit 2",
]

RENAMED_COLUMNS = [
    "concept_id",
    "lang_1",
    "word_1",
    "lang_2",
    "word_2",
    "translit_1",
    "translit_2",
]


# =========================================================
# Normalization
# =========================================================
_WS_RE = re.compile(r"\s+")


def normalize_text(value: Any, *, lowercase: bool = True) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


def normalize_lookup(value: Any) -> str:
    return normalize_text(value, lowercase=True)


# =========================================================
# Chunk filtering
# =========================================================
def _normalize_chunk_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    if list(chunk.columns) != EXPECTED_RAW_COLUMNS:
        if len(chunk.columns) < 7:
            raise ValueError(
                f"CogNet chunk has fewer than 7 columns: {list(chunk.columns)}"
            )
        chunk = chunk.iloc[:, :7].copy()
        chunk.columns = EXPECTED_RAW_COLUMNS

    chunk = chunk.rename(
        columns={
            "concept id": "concept_id",
            "lang 1": "lang_1",
            "word 1": "word_1",
            "lang 2": "lang_2",
            "word 2": "word_2",
            "translit 1": "translit_1",
            "translit 2": "translit_2",
        }
    ).copy()

    for col in RENAMED_COLUMNS:
        chunk[col] = chunk[col].fillna("").astype(str)

    chunk["lang_1"] = chunk["lang_1"].str.strip().str.lower()
    chunk["lang_2"] = chunk["lang_2"].str.strip().str.lower()
    return chunk


def _filter_relevant_rows(chunk: pd.DataFrame) -> pd.DataFrame:
    lang_1 = chunk["lang_1"]
    lang_2 = chunk["lang_2"]

    mask = (
        (lang_1.isin(BEA_SOURCE_LANGS) & (lang_2 == TARGET_LANG))
        | (lang_2.isin(BEA_SOURCE_LANGS) & (lang_1 == TARGET_LANG))
    )
    return chunk.loc[mask].copy()


def _add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_1_norm"] = df["word_1"].map(normalize_lookup)
    df["word_2_norm"] = df["word_2"].map(normalize_lookup)
    df["translit_1_norm"] = df["translit_1"].map(normalize_lookup)
    df["translit_2_norm"] = df["translit_2"].map(normalize_lookup)
    return df


def prepare_subset(
    cognet_path: Path,
    *,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bad_lines: list[list[str]] = []
    kept_chunks: list[pd.DataFrame] = []

    def _handle_bad_line(fields: list[str]) -> None:
        bad_lines.append(fields)
        return None

    reader = pd.read_csv(
        cognet_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines=_handle_bad_line,
        chunksize=chunksize,
        usecols=range(7),
        encoding="utf-8",
    )

    n_chunks = 0
    n_rows_seen = 0
    n_rows_kept = 0

    for chunk in reader:
        n_chunks += 1
        n_rows_seen += len(chunk)

        chunk = _normalize_chunk_columns(chunk)
        chunk = _filter_relevant_rows(chunk)

        if chunk.empty:
            continue

        n_rows_kept += len(chunk)
        kept_chunks.append(chunk)

    if kept_chunks:
        subset = pd.concat(kept_chunks, ignore_index=True)
    else:
        subset = pd.DataFrame(columns=RENAMED_COLUMNS)

    subset = _add_normalized_columns(subset)

    stats = {
        "cognet_path": str(cognet_path),
        "n_chunks": int(n_chunks),
        "n_rows_seen": int(n_rows_seen),
        "n_rows_kept": int(n_rows_kept),
        "n_bad_lines_skipped": int(len(bad_lines)),
        "source_langs_kept": sorted(BEA_SOURCE_LANGS),
        "target_lang_kept": TARGET_LANG,
        "rows_by_lang_pair": {
            f"{l1}->{l2}": int(count)
            for (l1, l2), count in subset.groupby(["lang_1", "lang_2"]).size().to_dict().items()
        } if not subset.empty else {},
    }
    return subset, stats


# =========================================================
# Saving
# =========================================================
def save_subset(df: pd.DataFrame, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return "parquet"
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return "csv"

    raise ValueError(
        f"Unsupported output format '{output_path.suffix}'. Use .parquet or .csv"
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compact CogNet subset for BEA languages (es/de/cn -> eng)."
    )
    parser.add_argument(
        "--cognet-path",
        type=Path,
        required=True,
        help="Path to the raw CogNet-v2.0.tsv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output subset path. Recommended: .parquet",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        required=True,
        help="JSON stats output path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for reading the raw TSV. Default: 200000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subset_df, stats = prepare_subset(
        args.cognet_path,
        chunksize=args.chunksize,
    )

    save_format = save_subset(subset_df, args.output)

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    summary = {
        "input": str(args.cognet_path),
        "output": str(args.output),
        "output_format": save_format,
        "stats_output": str(args.stats_output),
        "n_subset_rows": int(len(subset_df)),
        "n_unique_concepts": int(subset_df["concept_id"].nunique()) if not subset_df.empty else 0,
        "n_unique_lang_1": int(subset_df["lang_1"].nunique()) if not subset_df.empty else 0,
        "n_unique_lang_2": int(subset_df["lang_2"].nunique()) if not subset_df.empty else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()