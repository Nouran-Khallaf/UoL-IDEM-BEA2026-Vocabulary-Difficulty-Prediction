from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd


@dataclass(slots=True)
class DiagnosticsUpdateResult:
    input_csv: Path
    input_diagnostics: Path
    output_diagnostics: Path
    n_original_feature_columns: int
    n_new_embedding_columns: int
    n_final_feature_columns: int
    added_embedding_columns: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read diagnostics JSON from {path}: {e}") from e


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception as e:
        raise ValueError(f"Failed to read CSV header from {csv_path}: {e}") from e
    return list(df.columns)


def _resolve_embedding_columns(
    csv_columns: list[str],
    *,
    prefix: str,
) -> list[str]:
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    resolved_prefix = prefix.strip()
    emb_cols = [c for c in csv_columns if c.startswith(f"{resolved_prefix}_")]

    if not emb_cols:
        raise ValueError(
            f"No embedding columns found with prefix '{resolved_prefix}_'."
        )

    def _suffix_as_int(name: str) -> tuple[int, str]:
        suffix = name[len(resolved_prefix) + 1 :]
        try:
            return int(suffix), name
        except ValueError:
            return 10**12, name

    emb_cols = sorted(emb_cols, key=_suffix_as_int)
    return emb_cols


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            output.append(v)
    return output


def update_feature_diagnostics_with_embeddings(
    *,
    input_csv: str | Path,
    input_diagnostics: str | Path,
    output_diagnostics: str | Path | None = None,
    prefix: str,
    update_n_features: bool = True,
    add_embedding_metadata: bool = True,
) -> DiagnosticsUpdateResult:
    """
    Update a processed-feature diagnostics JSON file so embedding columns are
    included in `feature_columns`.

    Parameters
    ----------
    input_csv:
        Embedding-augmented CSV path.
    input_diagnostics:
        Original diagnostics JSON path.
    output_diagnostics:
        Output diagnostics JSON path. If None, overwrite input_diagnostics.
    prefix:
        Embedding column prefix, e.g. 'labse', 'xlmr', 'mbert'.
    update_n_features:
        If True, update simple count fields when they exist.
    add_embedding_metadata:
        If True, store an `embedding_features` metadata block.

    Returns
    -------
    DiagnosticsUpdateResult
    """
    input_csv = Path(input_csv)
    input_diagnostics = Path(input_diagnostics)
    output_diagnostics = Path(output_diagnostics) if output_diagnostics is not None else input_diagnostics

    diagnostics = _read_json(input_diagnostics)
    csv_columns = _load_columns(input_csv)
    embedding_columns = _resolve_embedding_columns(csv_columns, prefix=prefix)

    feature_columns = diagnostics.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            f"Diagnostics file {input_diagnostics} does not contain a valid non-empty 'feature_columns' list."
        )

    feature_columns = [str(c).strip() for c in feature_columns if str(c).strip()]
    merged_feature_columns = _deduplicate_preserve_order(feature_columns + embedding_columns)

    updated = dict(diagnostics)
    updated["feature_columns"] = merged_feature_columns

    if update_n_features:
        for key in ("n_features", "num_features", "feature_count", "n_feature_columns"):
            if key in updated:
                updated[key] = int(len(merged_feature_columns))

    if add_embedding_metadata:
        updated["embedding_features"] = {
            "enabled": True,
            "prefix": prefix,
            "n_embedding_features": int(len(embedding_columns)),
            "embedding_columns": embedding_columns,
        }

    _write_json(updated, output_diagnostics)

    return DiagnosticsUpdateResult(
        input_csv=input_csv,
        input_diagnostics=input_diagnostics,
        output_diagnostics=output_diagnostics,
        n_original_feature_columns=int(len(feature_columns)),
        n_new_embedding_columns=int(len(embedding_columns)),
        n_final_feature_columns=int(len(merged_feature_columns)),
        added_embedding_columns=embedding_columns,
    )


def clone_and_update_train_dev_diagnostics(
    *,
    feature_dir: str | Path,
    train_csv: str,
    dev_csv: str,
    train_diagnostics: str,
    dev_diagnostics: str,
    output_train_diagnostics: str,
    output_dev_diagnostics: str,
    prefix: str,
) -> dict[str, DiagnosticsUpdateResult]:
    """
    Convenience helper for updating both train/dev diagnostics in one call.
    """
    feature_dir = Path(feature_dir)

    train_result = update_feature_diagnostics_with_embeddings(
        input_csv=feature_dir / train_csv,
        input_diagnostics=feature_dir / train_diagnostics,
        output_diagnostics=feature_dir / output_train_diagnostics,
        prefix=prefix,
    )

    dev_result = update_feature_diagnostics_with_embeddings(
        input_csv=feature_dir / dev_csv,
        input_diagnostics=feature_dir / dev_diagnostics,
        output_diagnostics=feature_dir / output_dev_diagnostics,
        prefix=prefix,
    )

    return {
        "train": train_result,
        "dev": dev_result,
    }