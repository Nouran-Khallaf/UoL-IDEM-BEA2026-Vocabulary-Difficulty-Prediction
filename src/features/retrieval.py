from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class RetrievalFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    target_word_column: str
    clue_column: str | None
    context_column: str | None
    candidate_count_column: str | None
    output_prefix: str
    compute_mode: str
    forbid_negative_values: list[str]
    clip_probability_to_unit_interval: bool


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}


# -------------------------------------------------
# Basic helpers
# -------------------------------------------------
def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Retrieval feature builder expects a pandas DataFrame.")
    return df



def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureValidationError(f"'{name}' must be a non-empty string.")
    return value.strip()



def _get_nested_dict(d: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    value = d.get(key, {})
    return value if isinstance(value, dict) else {}



def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception as e:
        raise FeatureValidationError(
            f"Failed to coerce retrieval column '{column_name}' to numeric: {e}"
        ) from e



def _cast_column(series: pd.Series, target_type: str, column_name: str) -> pd.Series:
    target_type = str(target_type).strip().lower()

    if target_type in {"float", "double"}:
        return _coerce_numeric(series, column_name).astype(float)

    if target_type in {"int", "integer"}:
        numeric = _coerce_numeric(series, column_name)
        if numeric.isna().any():
            bad_count = int(numeric.isna().sum())
            raise FeatureValidationError(
                f"Column '{column_name}' cannot be cast to int because it has {bad_count} missing/non-numeric values after coercion."
            )
        return numeric.astype(int)

    if target_type in {"str", "string"}:
        return series.astype("string")

    raise FeatureValidationError(
        f"Unsupported retrieval cast type '{target_type}' for column '{column_name}'."
    )



def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


# -------------------------------------------------
# Config parsing
# -------------------------------------------------
def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> RetrievalFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for retrieval features must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for retrieval features must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("retrieval.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("retrieval.validation must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "retrieval")).strip()
    if not output_prefix:
        raise FeatureValidationError("retrieval.output_prefix must be a non-empty string.")

    default_columns_expected = [
        f"{output_prefix}_target_in_clue",
        f"{output_prefix}_target_in_context",
        f"{output_prefix}_candidate_count",
        f"{output_prefix}_target_prior",
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_columns_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("retrieval.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {
        f"{output_prefix}_target_in_clue": 0,
        f"{output_prefix}_target_in_context": 0,
        f"{output_prefix}_candidate_count": 0,
        f"{output_prefix}_target_prior": 0.0,
    }
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {
        f"{output_prefix}_target_in_clue": "int",
        f"{output_prefix}_target_in_context": "int",
        f"{output_prefix}_candidate_count": "int",
        f"{output_prefix}_target_prior": "float",
    }
    cast.update(_get_nested_dict(preprocessing, "cast"))

    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    clue_column = feature_group_cfg.get("clue_column") or columns_cfg.get("clue")
    context_column = feature_group_cfg.get("context_column") or columns_cfg.get("context")
    candidate_count_column = feature_group_cfg.get("candidate_count_column") or columns_cfg.get("candidate_count")

    target_word_column = _require_nonempty_string(target_word_column, "retrieval.target_word_column")
    if clue_column is not None:
        clue_column = _require_nonempty_string(clue_column, "retrieval.clue_column")
    if context_column is not None:
        context_column = _require_nonempty_string(context_column, "retrieval.context_column")
    if candidate_count_column is not None:
        candidate_count_column = _require_nonempty_string(candidate_count_column, "retrieval.candidate_count_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported retrieval.compute_mode '{compute_mode}'. Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    forbid_negative_values = validation.get(
        "forbid_negative_values",
        [f"{output_prefix}_candidate_count", f"{output_prefix}_target_prior"],
    )
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("retrieval.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    clip_probability_to_unit_interval = bool(validation.get("clip_probability_to_unit_interval", True))

    return RetrievalFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        target_word_column=target_word_column,
        clue_column=clue_column,
        context_column=context_column,
        candidate_count_column=candidate_count_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        forbid_negative_values=forbid_negative_values,
        clip_probability_to_unit_interval=clip_probability_to_unit_interval,
    )


# -------------------------------------------------
# Feature computation
# -------------------------------------------------
def _target_in_text(target: Any, text: Any) -> int:
    tgt = _normalized_text(target)
    txt = _normalized_text(text)
    if not tgt or not txt:
        return 0
    return int(tgt in txt)



def _safe_candidate_count(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0



def _pseudo_target_prior(target: Any, clue: Any, context: Any, candidate_count: int) -> float:
    tgt = _normalized_text(target)
    if not tgt:
        return 0.0

    score = 0.05
    score += 0.25 * _target_in_text(tgt, clue)
    score += 0.20 * _target_in_text(tgt, context)

    if candidate_count > 0:
        score += min(0.35, 1.0 / candidate_count)
    else:
        score += 0.05

    score += 0.05 / (1.0 + max(0, len(tgt) - 5))
    return float(max(0.0, min(1.0, score)))



def _compute_retrieval_features(df: pd.DataFrame, cfg: RetrievalFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    in_clue_col = f"{cfg.output_prefix}_target_in_clue"
    in_context_col = f"{cfg.output_prefix}_target_in_context"
    cand_count_col = f"{cfg.output_prefix}_candidate_count"
    prior_col = f"{cfg.output_prefix}_target_prior"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.target_word_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute retrieval features: missing target word column '{cfg.target_word_column}'."
        )

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in (in_clue_col, in_context_col, cand_count_col, prior_col) if c in result.columns]
        if len(existing) == 4 and all(not result[c].isna().all() for c in existing):
            return result

    target_series = result[cfg.target_word_column]
    clue_series = result[cfg.clue_column] if cfg.clue_column and cfg.clue_column in result.columns else pd.Series([None] * len(result), index=result.index)
    context_series = result[cfg.context_column] if cfg.context_column and cfg.context_column in result.columns else pd.Series([None] * len(result), index=result.index)

    if cfg.candidate_count_column and cfg.candidate_count_column in result.columns:
        candidate_counts = [_safe_candidate_count(v) for v in result[cfg.candidate_count_column]]
    else:
        candidate_counts = [
            max(1, _target_in_text(tgt, clue) + _target_in_text(tgt, ctx))
            for tgt, clue, ctx in zip(target_series, clue_series, context_series)
        ]

    result[in_clue_col] = [_target_in_text(tgt, clue) for tgt, clue in zip(target_series, clue_series)]
    result[in_context_col] = [_target_in_text(tgt, ctx) for tgt, ctx in zip(target_series, context_series)]
    result[cand_count_col] = candidate_counts
    result[prior_col] = [
        _pseudo_target_prior(tgt, clue, ctx, cnt)
        for tgt, clue, ctx, cnt in zip(target_series, clue_series, context_series, candidate_counts)
    ]
    return result


# -------------------------------------------------
# Validation / postprocessing
# -------------------------------------------------
def _fill_missing_columns(df: pd.DataFrame, cfg: RetrievalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result



def _cast_columns(df: pd.DataFrame, cfg: RetrievalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result



def _clip_probabilities(df: pd.DataFrame, cfg: RetrievalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    if not cfg.clip_probability_to_unit_interval:
        return result

    prior_col = f"{cfg.output_prefix}_target_prior"
    if prior_col in result.columns:
        numeric = pd.to_numeric(result[prior_col], errors="coerce")
        result[prior_col] = numeric.clip(lower=0.0, upper=1.0)
    return result



def _ensure_expected_columns(df: pd.DataFrame, cfg: RetrievalFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Retrieval feature group is missing expected columns in split '{split_name}': {missing}"
        )



def _validate_negative_constraints(df: pd.DataFrame, cfg: RetrievalFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"Retrieval column '{col}' contains negative values in split '{split_name}'."
            )



def _collect_feature_columns(df: pd.DataFrame, cfg: RetrievalFeatureConfig) -> list[str]:
    cols: list[str] = []
    for col in cfg.columns_expected:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


# -------------------------------------------------
# Public API
# -------------------------------------------------
def build_features(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    feature_group_cfg: dict[str, Any] | None = None,
    split_name: str = "unknown",
) -> pd.DataFrame:
    """
    Build/validate retrieval-style features.

    This module computes lightweight retrieval-oriented signals from the target
    word, clue, context, and optional candidate-count column:
      - retrieval_target_in_clue
      - retrieval_target_in_context
      - retrieval_candidate_count
      - retrieval_target_prior

    It is computation-first and does not assume the retrieval features already exist.
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_retrieval_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)
    result = _clip_probabilities(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    retrieval_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["retrieval_feature_columns"] = retrieval_feature_columns
    return result



def build_feature_group(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    feature_group_cfg: dict[str, Any] | None = None,
    split_name: str = "unknown",
) -> pd.DataFrame:
    return build_features(
        df,
        cfg=cfg,
        feature_group_cfg=feature_group_cfg,
        split_name=split_name,
    )



def build(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)
