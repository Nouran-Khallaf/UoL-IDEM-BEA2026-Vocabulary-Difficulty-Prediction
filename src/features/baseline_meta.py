from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class BaselineMetaFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    prediction_column: str | None
    target_word_column: str | None
    clue_column: str | None
    output_prefix: str
    compute_mode: str
    forbid_negative_values: list[str]


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Baseline-meta feature builder expects a pandas DataFrame.")
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
            f"Failed to coerce baseline-meta column '{column_name}' to numeric: {e}"
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
                f"Column '{column_name}' cannot be cast to int because it has "
                f"{bad_count} missing/non-numeric values after coercion."
            )
        return numeric.astype(int)

    if target_type in {"str", "string"}:
        return series.astype("string")

    raise FeatureValidationError(
        f"Unsupported baseline-meta cast type '{target_type}' for column '{column_name}'."
    )


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> BaselineMetaFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for baseline_meta must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for baseline_meta must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("baseline_meta.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("baseline_meta.validation must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "baseline")).strip()
    if not output_prefix:
        raise FeatureValidationError("baseline_meta.output_prefix must be a non-empty string.")

    default_columns_expected = [
        f"{output_prefix}_pred_len",
        f"{output_prefix}_pred_matches_target",
        f"{output_prefix}_clue_overlap",
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_columns_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("baseline_meta.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {
        f"{output_prefix}_pred_len": 0,
        f"{output_prefix}_pred_matches_target": 0,
        f"{output_prefix}_clue_overlap": 0.0,
    }
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {
        f"{output_prefix}_pred_len": "int",
        f"{output_prefix}_pred_matches_target": "int",
        f"{output_prefix}_clue_overlap": "float",
    }
    cast.update(_get_nested_dict(preprocessing, "cast"))

    prediction_column = feature_group_cfg.get("prediction_column") or columns_cfg.get("predicted_word")
    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    clue_column = feature_group_cfg.get("clue_column") or columns_cfg.get("clue")

    if prediction_column is not None:
        prediction_column = _require_nonempty_string(prediction_column, "baseline_meta.prediction_column")
    if target_word_column is not None:
        target_word_column = _require_nonempty_string(target_word_column, "baseline_meta.target_word_column")
    if clue_column is not None:
        clue_column = _require_nonempty_string(clue_column, "baseline_meta.clue_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported baseline_meta.compute_mode '{compute_mode}'. "
            f"Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    forbid_negative_values = validation.get(
        "forbid_negative_values",
        [f"{output_prefix}_pred_len", f"{output_prefix}_clue_overlap"],
    )
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("baseline_meta.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    return BaselineMetaFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        prediction_column=prediction_column,
        target_word_column=target_word_column,
        clue_column=clue_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        forbid_negative_values=forbid_negative_values,
    )


def _prediction_length(value: Any) -> int:
    return len(_normalized_text(value))


def _match_target(prediction: Any, target: Any) -> int:
    pred = _normalized_text(prediction)
    tgt = _normalized_text(target)
    return int(pred == tgt and tgt != "")


def _clue_overlap(prediction: Any, clue: Any) -> float:
    pred = _normalized_text(prediction)
    clue_text = _normalized_text(clue)
    if not pred or not clue_text:
        return 0.0

    pred_chars = set(pred)
    clue_chars = set(clue_text)
    union = pred_chars | clue_chars
    if not union:
        return 0.0
    return float(len(pred_chars & clue_chars) / len(union))


def _compute_baseline_meta_features(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    pred_len_col = f"{cfg.output_prefix}_pred_len"
    match_col = f"{cfg.output_prefix}_pred_matches_target"
    clue_overlap_col = f"{cfg.output_prefix}_clue_overlap"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.prediction_column is None or cfg.prediction_column not in result.columns:
        if cfg.compute_mode == "always_compute":
            result[pred_len_col] = 0
            result[match_col] = 0
            result[clue_overlap_col] = 0.0
        return result

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in (pred_len_col, match_col, clue_overlap_col) if c in result.columns]
        if len(existing) == 3 and all(not result[c].isna().all() for c in existing):
            return result

    pred_series = result[cfg.prediction_column]
    target_series = (
        result[cfg.target_word_column]
        if cfg.target_word_column and cfg.target_word_column in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )
    clue_series = (
        result[cfg.clue_column]
        if cfg.clue_column and cfg.clue_column in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )

    result[pred_len_col] = [_prediction_length(v) for v in pred_series]
    result[match_col] = [_match_target(pred, tgt) for pred, tgt in zip(pred_series, target_series)]
    result[clue_overlap_col] = [_clue_overlap(pred, clue) for pred, clue in zip(pred_series, clue_series)]
    return result


def _fill_missing_columns(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result


def _cast_columns(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result


def _ensure_expected_columns(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"baseline_meta feature group is missing expected columns in split '{split_name}': {missing}"
        )


def _validate_negative_constraints(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"baseline_meta column '{col}' contains negative values in split '{split_name}'."
            )


def _collect_feature_columns(df: pd.DataFrame, cfg: BaselineMetaFeatureConfig) -> list[str]:
    cols: list[str] = []
    for col in cfg.columns_expected:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def build_features(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    feature_group_cfg: dict[str, Any] | None = None,
    split_name: str = "unknown",
) -> pd.DataFrame:
    """
    Build/validate baseline-meta features over an existing predicted-word column.

    Produced columns (with output_prefix='baseline'):
      - baseline_pred_len
      - baseline_pred_matches_target
      - baseline_clue_overlap
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_baseline_meta_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["baseline_meta_feature_columns"] = feature_columns
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