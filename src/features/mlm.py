from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class MLMFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    target_word_column: str
    context_column: str
    clue_column: str | None
    output_prefix: str
    compute_mode: str
    model_name: str
    device: str | None
    max_length: int
    clip_rank_min: int
    forbid_negative_values: list[str]


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}
_MODEL_CACHE: dict[tuple[str, str], tuple[AutoTokenizer, AutoModelForMaskedLM]] = {}


# -------------------------------------------------
# Basic helpers
# -------------------------------------------------
def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("MLM feature builder expects a pandas DataFrame.")
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
            f"Failed to coerce MLM column '{column_name}' to numeric: {e}"
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
        f"Unsupported MLM cast type '{target_type}' for column '{column_name}'."
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
            f"Failed to load MLM model/tokenizer '{model_name}': {e}"
        ) from e

    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


# -------------------------------------------------
# Config parsing
# -------------------------------------------------
def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> MLMFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for MLM features must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for MLM features must be a dictionary when provided.")

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
        raise FeatureValidationError("mlm.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("mlm.validation must be a dictionary.")
    if not isinstance(model_cfg, dict):
        raise FeatureValidationError("mlm.model must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "mlm")).strip()
    if not output_prefix:
        raise FeatureValidationError("mlm.output_prefix must be a non-empty string.")

    default_columns_expected = [
        f"{output_prefix}_log_prob",
        f"{output_prefix}_rank",
        f"{output_prefix}_entropy",
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_columns_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("mlm.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {
        f"{output_prefix}_log_prob": 0.0,
        f"{output_prefix}_rank": 0,
        f"{output_prefix}_entropy": 0.0,
    }
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {
        f"{output_prefix}_log_prob": "float",
        f"{output_prefix}_rank": "int",
        f"{output_prefix}_entropy": "float",
    }
    cast.update(_get_nested_dict(preprocessing, "cast"))

    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    context_column = feature_group_cfg.get("context_column") or columns_cfg.get("context") or "context"
    clue_column = feature_group_cfg.get("clue_column") or columns_cfg.get("clue")

    target_word_column = _require_nonempty_string(target_word_column, "mlm.target_word_column")
    context_column = _require_nonempty_string(context_column, "mlm.context_column")
    if clue_column is not None:
        clue_column = _require_nonempty_string(clue_column, "mlm.clue_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported mlm.compute_mode '{compute_mode}'. Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    model_name = model_cfg.get("name") or feature_group_cfg.get("model_name") or "bert-base-multilingual-cased"
    model_name = _require_nonempty_string(model_name, "mlm.model.name")

    device = model_cfg.get("device") or runtime_cfg.get("device")
    if device is not None:
        device = _require_nonempty_string(device, "mlm.model.device")

    max_length = model_cfg.get("max_length", 256)
    if not isinstance(max_length, int) or max_length < 8:
        raise FeatureValidationError("mlm.model.max_length must be an integer >= 8.")

    clip_rank_min = validation.get("clip_rank_min", 1)
    if not isinstance(clip_rank_min, int) or clip_rank_min < 1:
        raise FeatureValidationError("mlm.validation.clip_rank_min must be an integer >= 1.")

    forbid_negative_values = validation.get(
        "forbid_negative_values",
        [f"{output_prefix}_rank", f"{output_prefix}_entropy"],
    )
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("mlm.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    return MLMFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        target_word_column=target_word_column,
        context_column=context_column,
        clue_column=clue_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        model_name=model_name,
        device=device,
        max_length=max_length,
        clip_rank_min=clip_rank_min,
        forbid_negative_values=forbid_negative_values,
    )


# -------------------------------------------------
# Real masked-LM scoring
# -------------------------------------------------
def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()



def _build_masked_input(
    *,
    tokenizer: AutoTokenizer,
    context: str,
    target_word: str,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    target_word = target_word.strip()
    if not target_word:
        raise FeatureValidationError("Target word is empty; cannot compute MLM features.")

    target_pieces = tokenizer.tokenize(target_word)
    if not target_pieces:
        raise FeatureValidationError(
            f"Tokenizer produced no wordpieces for target word: '{target_word}'"
        )

    n_masks = len(target_pieces)
    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise FeatureValidationError("Tokenizer does not define a mask token.")

    masked_suffix = " ".join([mask_token] * n_masks)
    text = f"{context} {tokenizer.sep_token} {masked_suffix}" if tokenizer.sep_token else f"{context} {masked_suffix}"

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"][0].tolist()
    mask_token_id = tokenizer.mask_token_id
    mask_positions = [i for i, tok_id in enumerate(input_ids) if tok_id == mask_token_id]

    if len(mask_positions) != n_masks:
        raise FeatureValidationError(
            f"Expected {n_masks} mask positions but found {len(mask_positions)} after tokenization/truncation."
        )

    target_ids = tokenizer.convert_tokens_to_ids(target_pieces)
    return encoded, mask_positions, target_ids



def _compute_entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    return float(entropy)



def _score_target_word(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    context: str,
    target_word: str,
    max_length: int,
) -> tuple[float, int, float]:
    encoded, mask_positions, target_ids = _build_masked_input(
        tokenizer=tokenizer,
        context=context,
        target_word=target_word,
        max_length=max_length,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0]

    log_probs = []
    ranks = []
    entropies = []

    for pos, target_id in zip(mask_positions, target_ids):
        pos_logits = logits[pos]
        pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
        target_log_prob = pos_log_probs[target_id].item()
        rank = int((pos_logits > pos_logits[target_id]).sum().item()) + 1
        entropy = _compute_entropy_from_logits(pos_logits)

        log_probs.append(float(target_log_prob))
        ranks.append(rank)
        entropies.append(entropy)

    total_log_prob = float(sum(log_probs))
    mean_rank = int(round(float(np.mean(ranks)))) if ranks else 0
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    return total_log_prob, mean_rank, mean_entropy



def _compute_mlm_features(df: pd.DataFrame, cfg: MLMFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    log_prob_col = f"{cfg.output_prefix}_log_prob"
    rank_col = f"{cfg.output_prefix}_rank"
    entropy_col = f"{cfg.output_prefix}_entropy"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.target_word_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute MLM features: missing target word column '{cfg.target_word_column}'."
        )
    if cfg.context_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute MLM features: missing context column '{cfg.context_column}'."
        )

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in (log_prob_col, rank_col, entropy_col) if c in result.columns]
        if len(existing) == 3 and all(not result[c].isna().all() for c in existing):
            return result

    device = _resolve_device(cfg.device)
    tokenizer, model = _get_model_and_tokenizer(cfg.model_name, device)

    log_probs: list[float] = []
    ranks: list[int] = []
    entropies: list[float] = []

    for _, row in result.iterrows():
        target_word = _safe_text(row[cfg.target_word_column])
        context = _safe_text(row[cfg.context_column])

        if not target_word or not context:
            log_probs.append(0.0)
            ranks.append(cfg.clip_rank_min)
            entropies.append(0.0)
            continue

        try:
            log_prob, rank, entropy = _score_target_word(
                tokenizer=tokenizer,
                model=model,
                device=device,
                context=context,
                target_word=target_word,
                max_length=cfg.max_length,
            )
        except Exception:
            log_prob, rank, entropy = 0.0, cfg.clip_rank_min, 0.0

        log_probs.append(log_prob)
        ranks.append(max(cfg.clip_rank_min, int(rank)))
        entropies.append(float(max(0.0, entropy)))

    result[log_prob_col] = log_probs
    result[rank_col] = ranks
    result[entropy_col] = entropies
    return result


# -------------------------------------------------
# Validation / postprocessing
# -------------------------------------------------
def _fill_missing_columns(df: pd.DataFrame, cfg: MLMFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result



def _cast_columns(df: pd.DataFrame, cfg: MLMFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result



def _ensure_expected_columns(df: pd.DataFrame, cfg: MLMFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"MLM feature group is missing expected columns in split '{split_name}': {missing}"
        )



def _validate_negative_constraints(df: pd.DataFrame, cfg: MLMFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"MLM column '{col}' contains negative values in split '{split_name}'."
            )



def _collect_feature_columns(df: pd.DataFrame, cfg: MLMFeatureConfig) -> list[str]:
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
    Build real masked-LM features for one split.

    Default model:
      - bert-base-multilingual-cased

    Change model in config, for example:
      model:
        name: xlm-roberta-base
        max_length: 256
        device: cuda

    Produced columns (with output_prefix='mlm'):
      - mlm_log_prob
      - mlm_rank
      - mlm_entropy
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_mlm_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    mlm_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["mlm_feature_columns"] = mlm_feature_columns
    result.attrs["mlm_model_name"] = parsed.model_name
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
