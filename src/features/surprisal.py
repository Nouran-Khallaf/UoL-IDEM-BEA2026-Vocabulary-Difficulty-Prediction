from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class SurprisalFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    target_word_column: str
    context_column: str
    output_prefix: str
    compute_mode: str
    model_name: str
    device: str | None
    max_length: int
    forbid_negative_values: list[str]
    compute_masked: bool
    compute_pll: bool
    aggregate_subwords: str


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}
_ALLOWED_AGGREGATES = {"mean", "sum"}
_MODEL_CACHE: dict[tuple[str, str], tuple[AutoTokenizer, AutoModelForMaskedLM]] = {}


# -------------------------------------------------
# Basic helpers
# -------------------------------------------------
def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Surprisal feature builder expects a pandas DataFrame.")
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
            f"Failed to coerce surprisal column '{column_name}' to numeric: {e}"
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

    raise FeatureValidationError(
        f"Unsupported surprisal cast type '{target_type}' for column '{column_name}'."
    )



def _resolve_device(device_str: str | None) -> str:
    if device_str is not None and device_str.strip():
        return device_str.strip()
    return "cuda" if torch.cuda.is_available() else "cpu"



def _get_model_and_tokenizer(model_name: str, device: str) -> tuple[AutoTokenizer, AutoModelForMaskedLM]:
    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    except Exception as e:
        raise FeatureValidationError(
            f"Failed to load surprisal model/tokenizer '{model_name}': {e}"
        ) from e

    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model



def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# -------------------------------------------------
# Config parsing
# -------------------------------------------------
def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> SurprisalFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for surprisal features must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for surprisal features must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})
    model_cfg = feature_group_cfg.get("model", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}
    runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if model_cfg is None:
        model_cfg = {}

    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("surprisal.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("surprisal.validation must be a dictionary.")
    if not isinstance(model_cfg, dict):
        raise FeatureValidationError("surprisal.model must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "surprisal")).strip()
    if not output_prefix:
        raise FeatureValidationError("surprisal.output_prefix must be a non-empty string.")

    compute_masked = bool(feature_group_cfg.get("compute_masked", True))
    compute_pll = bool(feature_group_cfg.get("compute_pll", True))
    aggregate_subwords = str(feature_group_cfg.get("aggregate_subwords", "mean")).strip().lower()
    if aggregate_subwords not in _ALLOWED_AGGREGATES:
        raise FeatureValidationError(
            f"Unsupported surprisal.aggregate_subwords '{aggregate_subwords}'. Allowed: {sorted(_ALLOWED_AGGREGATES)}"
        )

    default_columns = []
    if compute_masked:
        default_columns.append(f"{output_prefix}_masked")
    if compute_pll:
        default_columns.append(f"{output_prefix}_pll")
    default_columns.extend([
        f"{output_prefix}_subword_mean",
        f"{output_prefix}_subword_sum",
    ])

    columns_expected = feature_group_cfg.get("columns_expected", default_columns)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("surprisal.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {col: 0.0 for col in columns_expected}
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {col: "float" for col in columns_expected}
    cast.update(_get_nested_dict(preprocessing, "cast"))

    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    context_column = feature_group_cfg.get("context_column") or columns_cfg.get("context") or "context"
    target_word_column = _require_nonempty_string(target_word_column, "surprisal.target_word_column")
    context_column = _require_nonempty_string(context_column, "surprisal.context_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported surprisal.compute_mode '{compute_mode}'. Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    model_name = model_cfg.get("name") or feature_group_cfg.get("model_name") or "bert-base-multilingual-cased"
    model_name = _require_nonempty_string(model_name, "surprisal.model.name")

    device = model_cfg.get("device") or runtime_cfg.get("device")
    if device is not None:
        device = _require_nonempty_string(device, "surprisal.model.device")

    max_length = model_cfg.get("max_length", 256)
    if not isinstance(max_length, int) or max_length < 8:
        raise FeatureValidationError("surprisal.model.max_length must be an integer >= 8.")

    forbid_negative_values = validation.get("forbid_negative_values", columns_expected)
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("surprisal.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    return SurprisalFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        target_word_column=target_word_column,
        context_column=context_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        model_name=model_name,
        device=device,
        max_length=max_length,
        forbid_negative_values=forbid_negative_values,
        compute_masked=compute_masked,
        compute_pll=compute_pll,
        aggregate_subwords=aggregate_subwords,
    )


# -------------------------------------------------
# Real surprisal scoring
# -------------------------------------------------
def _build_masked_multi_input(
    *,
    tokenizer: AutoTokenizer,
    context: str,
    target_word: str,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        raise FeatureValidationError(f"Tokenizer produced no wordpieces for target word: '{target_word}'")

    n_masks = len(target_tokens)
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise FeatureValidationError("Tokenizer does not define a mask token.")

    masked_suffix = " ".join([mask_token] * n_masks)
    text = f"{context} {tokenizer.sep_token} {masked_suffix}" if tokenizer.sep_token else f"{context} {masked_suffix}"

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"][0].tolist()
    mask_positions = [i for i, tok_id in enumerate(input_ids) if tok_id == tokenizer.mask_token_id]
    if len(mask_positions) != n_masks:
        raise FeatureValidationError(
            f"Expected {n_masks} mask positions but found {len(mask_positions)} after tokenization/truncation."
        )

    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    return encoded, mask_positions, target_ids



def _build_pll_input(
    *,
    tokenizer: AutoTokenizer,
    context: str,
    target_word: str,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        raise FeatureValidationError(f"Tokenizer produced no wordpieces for target word: '{target_word}'")

    seq = f"{context} {tokenizer.sep_token} {target_word}" if tokenizer.sep_token else f"{context} {target_word}"
    encoded = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"][0].tolist()
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    positions = []
    # find last occurrence of target token sequence in encoded ids
    for start in range(0, len(input_ids) - len(target_ids) + 1):
        if input_ids[start:start + len(target_ids)] == target_ids:
            positions = list(range(start, start + len(target_ids)))
    if not positions:
        raise FeatureValidationError("Could not align target wordpieces inside PLL input.")

    return encoded, positions, target_ids



def _log_prob_for_token(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits, dim=-1)[token_id].item())



def _masked_surprisal(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    context: str,
    target_word: str,
    max_length: int,
    aggregate_subwords: str,
) -> tuple[float, list[float]]:
    encoded, mask_positions, target_ids = _build_masked_multi_input(
        tokenizer=tokenizer,
        context=context,
        target_word=target_word,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]

    subword_log_probs = []
    for pos, target_id in zip(mask_positions, target_ids):
        subword_log_probs.append(_log_prob_for_token(logits[pos], target_id))

    if aggregate_subwords == "sum":
        log_prob = float(sum(subword_log_probs))
    else:
        log_prob = float(np.mean(subword_log_probs))

    surprisal = float(-log_prob)
    return surprisal, subword_log_probs



def _pll_surprisal(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    context: str,
    target_word: str,
    max_length: int,
    aggregate_subwords: str,
) -> float:
    encoded, positions, target_ids = _build_pll_input(
        tokenizer=tokenizer,
        context=context,
        target_word=target_word,
        max_length=max_length,
    )

    base_input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    subword_log_probs = []

    for pos, target_id in zip(positions, target_ids):
        masked_ids = base_input_ids.clone()
        masked_ids[pos] = tokenizer.mask_token_id

        batch = {
            "input_ids": masked_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            logits = model(**batch).logits[0]
        subword_log_probs.append(_log_prob_for_token(logits[pos], target_id))

    if aggregate_subwords == "sum":
        log_prob = float(sum(subword_log_probs))
    else:
        log_prob = float(np.mean(subword_log_probs))
    return float(-log_prob)



def _compute_surprisal_features(df: pd.DataFrame, cfg: SurprisalFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    masked_col = f"{cfg.output_prefix}_masked"
    pll_col = f"{cfg.output_prefix}_pll"
    subword_mean_col = f"{cfg.output_prefix}_subword_mean"
    subword_sum_col = f"{cfg.output_prefix}_subword_sum"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.target_word_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute surprisal features: missing target word column '{cfg.target_word_column}'."
        )
    if cfg.context_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute surprisal features: missing context column '{cfg.context_column}'."
        )

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in cfg.columns_expected if c in result.columns]
        if len(existing) == len(cfg.columns_expected) and all(not result[c].isna().all() for c in existing):
            return result

    device = _resolve_device(cfg.device)
    tokenizer, model = _get_model_and_tokenizer(cfg.model_name, device)

    masked_vals: list[float] = []
    pll_vals: list[float] = []
    subword_mean_vals: list[float] = []
    subword_sum_vals: list[float] = []

    for _, row in result.iterrows():
        target_word = _safe_text(row[cfg.target_word_column])
        context = _safe_text(row[cfg.context_column])

        if not target_word or not context:
            masked_vals.append(0.0)
            pll_vals.append(0.0)
            subword_mean_vals.append(0.0)
            subword_sum_vals.append(0.0)
            continue

        try:
            if cfg.compute_masked:
                masked_surp, subword_log_probs = _masked_surprisal(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    context=context,
                    target_word=target_word,
                    max_length=cfg.max_length,
                    aggregate_subwords=cfg.aggregate_subwords,
                )
                masked_vals.append(masked_surp)
                subword_mean_vals.append(float(-np.mean(subword_log_probs)))
                subword_sum_vals.append(float(-np.sum(subword_log_probs)))
            else:
                masked_vals.append(0.0)
                subword_mean_vals.append(0.0)
                subword_sum_vals.append(0.0)

            if cfg.compute_pll:
                pll_vals.append(
                    _pll_surprisal(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        context=context,
                        target_word=target_word,
                        max_length=cfg.max_length,
                        aggregate_subwords=cfg.aggregate_subwords,
                    )
                )
            else:
                pll_vals.append(0.0)
        except Exception:
            masked_vals.append(0.0)
            pll_vals.append(0.0)
            subword_mean_vals.append(0.0)
            subword_sum_vals.append(0.0)

    if cfg.compute_masked:
        result[masked_col] = masked_vals
    if cfg.compute_pll:
        result[pll_col] = pll_vals
    result[subword_mean_col] = subword_mean_vals
    result[subword_sum_col] = subword_sum_vals
    return result


# -------------------------------------------------
# Validation / postprocessing
# -------------------------------------------------
def _fill_missing_columns(df: pd.DataFrame, cfg: SurprisalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result



def _cast_columns(df: pd.DataFrame, cfg: SurprisalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result



def _ensure_expected_columns(df: pd.DataFrame, cfg: SurprisalFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Surprisal feature group is missing expected columns in split '{split_name}': {missing}"
        )



def _validate_negative_constraints(df: pd.DataFrame, cfg: SurprisalFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"Surprisal column '{col}' contains negative values in split '{split_name}'."
            )



def _collect_feature_columns(df: pd.DataFrame, cfg: SurprisalFeatureConfig) -> list[str]:
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
    Build real surprisal features for one split.

    Main outputs
    ------------
    - surprisal_masked
    - surprisal_pll
    - surprisal_subword_mean
    - surprisal_subword_sum

    Design
    ------
    - masked variant: mask all target wordpieces simultaneously
    - PLL variant: mask one target wordpiece at a time
    - aggregation can be mean or sum over target subwords
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_surprisal_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    surprisal_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["surprisal_feature_columns"] = surprisal_feature_columns
    result.attrs["surprisal_model_name"] = parsed.model_name
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
