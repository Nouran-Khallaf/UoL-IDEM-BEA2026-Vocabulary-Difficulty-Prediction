from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class MLMPredictedWordConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]

    context_column: str
    clue_column: str | None
    target_word_column: str | None
    pos_column: str | None

    output_prefix: str
    compute_mode: str

    model_name: str
    device: str | None
    max_length: int
    batch_size: int

    candidate_source: str
    candidate_column: str | None
    max_candidates: int | None
    shortlist_top_k: int | None

    use_first_letter_constraint: bool
    use_length_constraint: bool
    length_tolerance: int

    use_pos_constraint: bool
    use_clue_pattern_constraint: bool


@dataclass(slots=True)
class CandidateEntry:
    word: str
    pos: str
    freq: float


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}
_MODEL_CACHE: dict[tuple[str, str], tuple[AutoTokenizer, AutoModelForMaskedLM]] = {}
_CANDIDATE_CACHE: dict[str, list[CandidateEntry]] = {}


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("MLM predicted-word builder expects a pandas DataFrame.")
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
            f"Failed to coerce MLM predicted-word column '{column_name}' to numeric: {e}"
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
        f"Unsupported cast type '{target_type}' for column '{column_name}'."
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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize(value: Any) -> str:
    return _safe_text(value).lower()


def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> MLMPredictedWordConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for mlm_predicted_word must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for mlm_predicted_word must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    model_cfg = feature_group_cfg.get("model", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}
    runtime_cfg = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("mlm_predicted_word.preprocessing must be a dictionary.")
    if not isinstance(model_cfg, dict):
        raise FeatureValidationError("mlm_predicted_word.model must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "mlm_pred")).strip()
    if not output_prefix:
        raise FeatureValidationError("mlm_predicted_word.output_prefix must be a non-empty string.")

    default_columns_expected = [
        f"{output_prefix}_word",
        f"{output_prefix}_log_prob",
        f"{output_prefix}_margin",
        f"{output_prefix}_entropy",
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_columns_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("mlm_predicted_word.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {
        f"{output_prefix}_word": "",
        f"{output_prefix}_log_prob": 0.0,
        f"{output_prefix}_margin": 0.0,
        f"{output_prefix}_entropy": 0.0,
    }
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {
        f"{output_prefix}_word": "string",
        f"{output_prefix}_log_prob": "float",
        f"{output_prefix}_margin": "float",
        f"{output_prefix}_entropy": "float",
    }
    cast.update(_get_nested_dict(preprocessing, "cast"))

    context_column = feature_group_cfg.get("context_column") or columns_cfg.get("context") or "L1_context"
    clue_column = feature_group_cfg.get("clue_column") or columns_cfg.get("clue") or "en_target_clue"
    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    pos_column = feature_group_cfg.get("pos_column") or columns_cfg.get("pos") or "en_target_pos"

    context_column = _require_nonempty_string(context_column, "mlm_predicted_word.context_column")
    if clue_column is not None:
        clue_column = _require_nonempty_string(clue_column, "mlm_predicted_word.clue_column")
    if target_word_column is not None:
        target_word_column = _require_nonempty_string(
            target_word_column, "mlm_predicted_word.target_word_column"
        )
    if pos_column is not None:
        pos_column = _require_nonempty_string(pos_column, "mlm_predicted_word.pos_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported mlm_predicted_word.compute_mode '{compute_mode}'. "
            f"Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    model_name = model_cfg.get("name") or feature_group_cfg.get("model_name") or "bert-base-multilingual-cased"
    model_name = _require_nonempty_string(model_name, "mlm_predicted_word.model.name")

    device = model_cfg.get("device") or runtime_cfg.get("device")
    if device is not None:
        device = _require_nonempty_string(device, "mlm_predicted_word.model.device")

    max_length = model_cfg.get("max_length", 256)
    if not isinstance(max_length, int) or max_length < 8:
        raise FeatureValidationError("mlm_predicted_word.model.max_length must be an integer >= 8.")

    batch_size = model_cfg.get("batch_size", feature_group_cfg.get("batch_size", 32))
    if not isinstance(batch_size, int) or batch_size < 1:
        raise FeatureValidationError("mlm_predicted_word.batch_size must be an integer >= 1.")

    candidate_source = str(feature_group_cfg.get("candidate_source", "target_column")).strip().lower()
    candidate_column = feature_group_cfg.get("candidate_column") or target_word_column
    if candidate_column is not None:
        candidate_column = _require_nonempty_string(candidate_column, "mlm_predicted_word.candidate_column")

    max_candidates = feature_group_cfg.get("max_candidates")
    if max_candidates is not None:
        if not isinstance(max_candidates, int) or max_candidates < 2:
            raise FeatureValidationError("mlm_predicted_word.max_candidates must be an integer >= 2.")

    shortlist_top_k = feature_group_cfg.get("shortlist_top_k", 100)
    if shortlist_top_k is not None:
        if not isinstance(shortlist_top_k, int) or shortlist_top_k < 1:
            raise FeatureValidationError("mlm_predicted_word.shortlist_top_k must be an integer >= 1.")

    use_first_letter_constraint = bool(feature_group_cfg.get("use_first_letter_constraint", True))
    use_length_constraint = bool(feature_group_cfg.get("use_length_constraint", True))

    length_tolerance = feature_group_cfg.get("length_tolerance", 0)
    if not isinstance(length_tolerance, int) or length_tolerance < 0:
        raise FeatureValidationError("mlm_predicted_word.length_tolerance must be an integer >= 0.")

    use_pos_constraint = bool(feature_group_cfg.get("use_pos_constraint", True))
    use_clue_pattern_constraint = bool(feature_group_cfg.get("use_clue_pattern_constraint", True))

    return MLMPredictedWordConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        context_column=context_column,
        clue_column=clue_column,
        target_word_column=target_word_column,
        pos_column=pos_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        candidate_source=candidate_source,
        candidate_column=candidate_column,
        max_candidates=max_candidates,
        shortlist_top_k=shortlist_top_k,
        use_first_letter_constraint=use_first_letter_constraint,
        use_length_constraint=use_length_constraint,
        length_tolerance=length_tolerance,
        use_pos_constraint=use_pos_constraint,
        use_clue_pattern_constraint=use_clue_pattern_constraint,
    )


def _get_candidate_vocabulary(df: pd.DataFrame, cfg: MLMPredictedWordConfig) -> list[CandidateEntry]:
    cache_key = (
        f"{cfg.candidate_source}::{cfg.candidate_column}::{cfg.pos_column}::"
        f"{cfg.max_candidates}"
    )
    if cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key]

    if cfg.candidate_column is None or cfg.candidate_column not in df.columns:
        raise FeatureValidationError(
            f"Cannot build candidate vocabulary: missing candidate column '{cfg.candidate_column}'."
        )

    candidate_series = df[cfg.candidate_column].dropna().astype(str).str.strip()
    candidate_series = candidate_series[candidate_series != ""]
    if candidate_series.empty:
        raise FeatureValidationError("Candidate vocabulary is empty for mlm_predicted_word.")

    freq_series = candidate_series.value_counts()

    pos_map: dict[str, str] = {}
    if cfg.pos_column is not None and cfg.pos_column in df.columns:
        tmp = df[[cfg.candidate_column, cfg.pos_column]].copy()
        tmp[cfg.candidate_column] = tmp[cfg.candidate_column].astype(str).str.strip()
        tmp[cfg.pos_column] = tmp[cfg.pos_column].astype(str).str.strip()
        tmp = tmp[(tmp[cfg.candidate_column] != "") & (tmp[cfg.pos_column] != "")]
        if not tmp.empty:
            mode_pos = tmp.groupby(cfg.candidate_column)[cfg.pos_column].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
            )
            pos_map = mode_pos.to_dict()

    vocab: list[CandidateEntry] = [
        CandidateEntry(
            word=word,
            pos=pos_map.get(word, ""),
            freq=float(freq),
        )
        for word, freq in freq_series.items()
    ]

    if cfg.max_candidates is not None:
        vocab = vocab[: cfg.max_candidates]

    if not vocab:
        raise FeatureValidationError("Candidate vocabulary is empty for mlm_predicted_word.")

    _CANDIDATE_CACHE[cache_key] = vocab
    return vocab


def _matches_clue_pattern(word: str, clue: str) -> bool:
    word_norm = _normalize(word)
    clue_norm = _normalize(clue)

    if not clue_norm:
        return True

    if len(word_norm) != len(clue_norm):
        return False

    for wc, cc in zip(word_norm, clue_norm):
        if cc == "_":
            continue
        if wc != cc:
            return False
    return True


def _filter_candidates(
    candidates: list[CandidateEntry],
    *,
    clue: str,
    target_word: str,
    row_pos: str,
    use_first_letter_constraint: bool,
    use_length_constraint: bool,
    length_tolerance: int,
    use_pos_constraint: bool,
    use_clue_pattern_constraint: bool,
) -> list[CandidateEntry]:
    filtered = candidates

    clue_norm = _normalize(clue)
    target_norm = _normalize(target_word)
    row_pos_norm = _normalize(row_pos)

    if use_first_letter_constraint and clue_norm:
        first_char = clue_norm[0]
        step = [c for c in filtered if _normalize(c.word).startswith(first_char)]
        filtered = step or filtered

    if use_clue_pattern_constraint and clue_norm:
        step = [c for c in filtered if _matches_clue_pattern(c.word, clue_norm)]
        filtered = step or filtered

    if use_length_constraint and target_norm:
        target_len = len(target_norm)
        step = [
            c for c in filtered
            if abs(len(_normalize(c.word)) - target_len) <= length_tolerance
        ]
        filtered = step or filtered
    elif use_length_constraint and clue_norm:
        clue_len = len(clue_norm)
        step = [
            c for c in filtered
            if abs(len(_normalize(c.word)) - clue_len) <= length_tolerance
        ]
        filtered = step or filtered

    if use_pos_constraint and row_pos_norm:
        step = [c for c in filtered if _normalize(c.pos) == row_pos_norm]
        filtered = step or filtered

    return filtered


def _shortlist_candidates(
    candidates: list[CandidateEntry],
    *,
    shortlist_top_k: int | None,
) -> list[CandidateEntry]:
    if shortlist_top_k is None or len(candidates) <= shortlist_top_k:
        return candidates

    ranked = sorted(candidates, key=lambda c: (-c.freq, _normalize(c.word)))
    return ranked[:shortlist_top_k]


def _build_masked_input(
    *,
    tokenizer: AutoTokenizer,
    context: str,
    target_word: str,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    target_pieces = tokenizer.tokenize(target_word)
    if not target_pieces:
        raise FeatureValidationError(f"Tokenizer produced no wordpieces for target word: '{target_word}'")

    mask_token = tokenizer.mask_token
    if mask_token is None:
        raise FeatureValidationError("Tokenizer does not define a mask token.")

    sep_token = tokenizer.sep_token or ""
    masked_suffix = " ".join([mask_token] * len(target_pieces))
    text = f"{context} {sep_token} {masked_suffix}".strip() if sep_token else f"{context} {masked_suffix}".strip()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"][0].tolist()
    mask_positions = [i for i, tok_id in enumerate(input_ids) if tok_id == tokenizer.mask_token_id]
    if len(mask_positions) != len(target_pieces):
        raise FeatureValidationError(
            f"Expected {len(target_pieces)} mask positions but found {len(mask_positions)}."
        )

    target_ids = tokenizer.convert_tokens_to_ids(target_pieces)
    return encoded, mask_positions, target_ids


def _score_candidates_batched(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    context: str,
    candidates: list[str],
    max_length: int,
    batch_size: int,
) -> list[tuple[str, float, float]]:
    prepared: list[
        tuple[str, torch.Tensor, torch.Tensor, list[int], list[int]]
    ] = []

    for cand in candidates:
        try:
            encoded, mask_positions, target_ids = _build_masked_input(
                tokenizer=tokenizer,
                context=context,
                target_word=cand,
                max_length=max_length,
            )
            prepared.append(
                (
                    cand,
                    encoded["input_ids"][0],
                    encoded["attention_mask"][0],
                    mask_positions,
                    target_ids,
                )
            )
        except Exception:
            continue

    if not prepared:
        return []

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token_id is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise FeatureValidationError("Tokenizer has no pad token, eos token, or sep token for batching.")

    scored: list[tuple[str, float, float]] = []

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start:start + batch_size]

        input_ids = pad_sequence(
            [item[1] for item in chunk],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item[2] for item in chunk],
            batch_first=True,
            padding_value=0,
        )

        batch = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }

        with torch.no_grad():
            logits = model(**batch).logits  # [B, T, V]

        for i, (cand, _, _, mask_positions, target_ids) in enumerate(chunk):
            row_logits = logits[i]

            log_probs: list[float] = []
            entropies: list[float] = []

            for pos, target_id in zip(mask_positions, target_ids):
                pos_logits = row_logits[pos]
                pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
                log_probs.append(float(pos_log_probs[target_id].item()))

                probs = torch.softmax(pos_logits, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
                entropies.append(float(entropy))

            scored.append(
                (
                    cand,
                    float(sum(log_probs)),
                    float(np.mean(entropies)) if entropies else 0.0,
                )
            )

    return scored


def _predict_word_for_row(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: str,
    context: str,
    clue: str,
    target_word: str,
    row_pos: str,
    candidates: list[CandidateEntry],
    cfg: MLMPredictedWordConfig,
) -> tuple[str, float, float, float]:
    usable_candidates = _filter_candidates(
        candidates,
        clue=clue,
        target_word=target_word,
        row_pos=row_pos,
        use_first_letter_constraint=cfg.use_first_letter_constraint,
        use_length_constraint=cfg.use_length_constraint,
        length_tolerance=cfg.length_tolerance,
        use_pos_constraint=cfg.use_pos_constraint,
        use_clue_pattern_constraint=cfg.use_clue_pattern_constraint,
    )

    usable_candidates = _shortlist_candidates(
        usable_candidates,
        shortlist_top_k=cfg.shortlist_top_k,
    )

    scored = _score_candidates_batched(
        tokenizer=tokenizer,
        model=model,
        device=device,
        context=context,
        candidates=[c.word for c in usable_candidates],
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )

    if not scored:
        return "", 0.0, 0.0, 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    best_word, best_log_prob, best_entropy = scored[0]
    second_log_prob = scored[1][1] if len(scored) > 1 else best_log_prob
    margin = float(best_log_prob - second_log_prob)

    return best_word, float(best_log_prob), margin, float(best_entropy)


def _compute_features(df: pd.DataFrame, cfg: MLMPredictedWordConfig) -> pd.DataFrame:
    result = df.copy()

    word_col = f"{cfg.output_prefix}_word"
    log_prob_col = f"{cfg.output_prefix}_log_prob"
    margin_col = f"{cfg.output_prefix}_margin"
    entropy_col = f"{cfg.output_prefix}_entropy"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.context_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute mlm_predicted_word features: missing context column '{cfg.context_column}'."
        )

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in (word_col, log_prob_col, margin_col, entropy_col) if c in result.columns]
        if len(existing) == 4 and all(not result[c].isna().all() for c in existing):
            return result

    device = _resolve_device(cfg.device)
    tokenizer, model = _get_model_and_tokenizer(cfg.model_name, device)
    candidates = _get_candidate_vocabulary(result, cfg)

    pred_words: list[str] = []
    pred_log_probs: list[float] = []
    pred_margins: list[float] = []
    pred_entropies: list[float] = []

    for _, row in result.iterrows():
        context = _safe_text(row[cfg.context_column])
        clue = _safe_text(row[cfg.clue_column]) if cfg.clue_column and cfg.clue_column in result.columns else ""
        target_word = _safe_text(row[cfg.target_word_column]) if cfg.target_word_column and cfg.target_word_column in result.columns else ""
        row_pos = _safe_text(row[cfg.pos_column]) if cfg.pos_column and cfg.pos_column in result.columns else ""

        if not context:
            pred_words.append("")
            pred_log_probs.append(0.0)
            pred_margins.append(0.0)
            pred_entropies.append(0.0)
            continue

        pred_word, pred_log_prob, pred_margin, pred_entropy = _predict_word_for_row(
            tokenizer=tokenizer,
            model=model,
            device=device,
            context=context,
            clue=clue,
            target_word=target_word,
            row_pos=row_pos,
            candidates=candidates,
            cfg=cfg,
        )
        pred_words.append(pred_word)
        pred_log_probs.append(pred_log_prob)
        pred_margins.append(pred_margin)
        pred_entropies.append(pred_entropy)

    result[word_col] = pred_words
    result[log_prob_col] = pred_log_probs
    result[margin_col] = pred_margins
    result[entropy_col] = pred_entropies
    return result


def _fill_missing_columns(df: pd.DataFrame, cfg: MLMPredictedWordConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result


def _cast_columns(df: pd.DataFrame, cfg: MLMPredictedWordConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result


def _ensure_expected_columns(df: pd.DataFrame, cfg: MLMPredictedWordConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"mlm_predicted_word feature group is missing expected columns in split '{split_name}': {missing}"
        )


def _collect_feature_columns(df: pd.DataFrame, cfg: MLMPredictedWordConfig) -> list[str]:
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
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    result.attrs["mlm_predicted_word_feature_columns"] = _collect_feature_columns(result, parsed)
    result.attrs["mlm_predicted_word_model_name"] = parsed.model_name
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