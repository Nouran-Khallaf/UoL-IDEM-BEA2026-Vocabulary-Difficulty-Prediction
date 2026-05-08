from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import pandas as pd

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class SemanticFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    source_word_column: str | None
    target_word_column: str | None
    source_domain_column: str | None
    target_domain_column: str | None
    source_domain_score_column: str | None
    target_domain_score_column: str | None
    clue_column: str | None
    context_column: str | None
    output_prefix: str
    compute_mode: str
    fallback_when_domains_missing: bool
    forbid_negative_values: list[str]
    clip_similarity_to_unit_interval: bool
    entropy_base: float


_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}


# -------------------------------------------------
# Basic helpers
# -------------------------------------------------
def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Semantic feature builder expects a pandas DataFrame.")
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
            f"Failed to coerce semantic column '{column_name}' to numeric: {e}"
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
        f"Unsupported semantic cast type '{target_type}' for column '{column_name}'."
    )



def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()



def _parse_domain_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        vals = value
    else:
        text = str(value).strip()
        if not text:
            return []
        for sep in (";", "|", ","):
            if sep in text:
                vals = [v.strip() for v in text.split(sep)]
                break
        else:
            vals = [text]
    cleaned = []
    for v in vals:
        vv = _normalized_text(v)
        if vv:
            cleaned.append(vv)
    return cleaned



def _parse_score_list(value: Any, n_items: int) -> list[float]:
    if n_items <= 0:
        return []
    if value is None:
        return [1.0] * n_items
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return [1.0] * n_items
        for sep in (";", "|", ","):
            if sep in text:
                raw = [v.strip() for v in text.split(sep)]
                break
        else:
            raw = [text]
    scores: list[float] = []
    for item in raw[:n_items]:
        try:
            scores.append(float(item))
        except Exception:
            scores.append(1.0)
    while len(scores) < n_items:
        scores.append(1.0)
    return scores



def _safe_entropy(probs: list[float], *, base: float) -> float:
    if not probs:
        return 0.0
    total = sum(max(0.0, p) for p in probs)
    if total <= 0.0:
        return 0.0
    norm = [max(0.0, p) / total for p in probs]
    entropy = 0.0
    for p in norm:
        if p > 0.0:
            entropy -= p * math.log(p, base)
    return float(entropy)



def _char_overlap(a: Any, b: Any) -> float:
    a_txt = _normalized_text(a)
    b_txt = _normalized_text(b)
    if not a_txt or not b_txt:
        return 0.0
    if a_txt == b_txt:
        return 1.0
    a_set = set(a_txt)
    b_set = set(b_txt)
    union = a_set | b_set
    if not union:
        return 0.0
    return float(len(a_set & b_set) / len(union))



def _token_overlap(a: Any, b: Any) -> float:
    a_txt = _normalized_text(a)
    b_txt = _normalized_text(b)
    if not a_txt or not b_txt:
        return 0.0
    a_tokens = set(a_txt.split())
    b_tokens = set(b_txt.split())
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return float(len(a_tokens & b_tokens) / len(union))


# -------------------------------------------------
# Config parsing
# -------------------------------------------------
def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> SemanticFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for semantic features must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for semantic features must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("semantic.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("semantic.validation must be a dictionary.")

    output_prefix = str(feature_group_cfg.get("output_prefix", "semantic")).strip()
    if not output_prefix:
        raise FeatureValidationError("semantic.output_prefix must be a non-empty string.")

    default_columns_expected = [
        f"{output_prefix}_usas_domain_match",
        f"{output_prefix}_usas_entropy_unweighted",
        f"{output_prefix}_usas_entropy_weighted",
        f"{output_prefix}_semantic_shift",
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_columns_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("semantic.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    fillna = {
        f"{output_prefix}_usas_domain_match": 0.0,
        f"{output_prefix}_usas_entropy_unweighted": 0.0,
        f"{output_prefix}_usas_entropy_weighted": 0.0,
        f"{output_prefix}_semantic_shift": 0.0,
    }
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = {
        f"{output_prefix}_usas_domain_match": "float",
        f"{output_prefix}_usas_entropy_unweighted": "float",
        f"{output_prefix}_usas_entropy_weighted": "float",
        f"{output_prefix}_semantic_shift": "float",
    }
    cast.update(_get_nested_dict(preprocessing, "cast"))

    source_word_column = feature_group_cfg.get("source_word_column") or columns_cfg.get("source_word")
    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word")
    clue_column = feature_group_cfg.get("clue_column") or columns_cfg.get("clue")
    context_column = feature_group_cfg.get("context_column") or columns_cfg.get("context")

    source_domain_column = feature_group_cfg.get("source_domain_column") or columns_cfg.get("source_domain")
    target_domain_column = feature_group_cfg.get("target_domain_column") or columns_cfg.get("target_domain")
    source_domain_score_column = feature_group_cfg.get("source_domain_score_column") or columns_cfg.get("source_domain_score")
    target_domain_score_column = feature_group_cfg.get("target_domain_score_column") or columns_cfg.get("target_domain_score")

    if source_word_column is not None:
        source_word_column = _require_nonempty_string(source_word_column, "semantic.source_word_column")
    if target_word_column is not None:
        target_word_column = _require_nonempty_string(target_word_column, "semantic.target_word_column")
    if clue_column is not None:
        clue_column = _require_nonempty_string(clue_column, "semantic.clue_column")
    if context_column is not None:
        context_column = _require_nonempty_string(context_column, "semantic.context_column")
    if source_domain_column is not None:
        source_domain_column = _require_nonempty_string(source_domain_column, "semantic.source_domain_column")
    if target_domain_column is not None:
        target_domain_column = _require_nonempty_string(target_domain_column, "semantic.target_domain_column")
    if source_domain_score_column is not None:
        source_domain_score_column = _require_nonempty_string(source_domain_score_column, "semantic.source_domain_score_column")
    if target_domain_score_column is not None:
        target_domain_score_column = _require_nonempty_string(target_domain_score_column, "semantic.target_domain_score_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported semantic.compute_mode '{compute_mode}'. Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    fallback_when_domains_missing = bool(feature_group_cfg.get("fallback_when_domains_missing", True))
    clip_similarity_to_unit_interval = bool(validation.get("clip_similarity_to_unit_interval", True))

    forbid_negative_values = validation.get(
        "forbid_negative_values",
        [
            f"{output_prefix}_usas_domain_match",
            f"{output_prefix}_usas_entropy_unweighted",
            f"{output_prefix}_usas_entropy_weighted",
            f"{output_prefix}_semantic_shift",
        ],
    )
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("semantic.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    entropy_base = validation.get("entropy_base", math.e)
    if not isinstance(entropy_base, (int, float)) or float(entropy_base) <= 1.0:
        raise FeatureValidationError("semantic.validation.entropy_base must be a number > 1.")

    return SemanticFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        source_word_column=source_word_column,
        target_word_column=target_word_column,
        source_domain_column=source_domain_column,
        target_domain_column=target_domain_column,
        source_domain_score_column=source_domain_score_column,
        target_domain_score_column=target_domain_score_column,
        clue_column=clue_column,
        context_column=context_column,
        output_prefix=output_prefix,
        compute_mode=compute_mode,
        fallback_when_domains_missing=fallback_when_domains_missing,
        forbid_negative_values=forbid_negative_values,
        clip_similarity_to_unit_interval=clip_similarity_to_unit_interval,
        entropy_base=float(entropy_base),
    )


# -------------------------------------------------
# Feature computation
# -------------------------------------------------
def _domain_match_score(source_domains: list[str], target_domains: list[str]) -> float:
    if not source_domains or not target_domains:
        return 0.0
    s = set(source_domains)
    t = set(target_domains)
    union = s | t
    if not union:
        return 0.0
    return float(len(s & t) / len(union))



def _semantic_shift_score(
    source_word: Any,
    target_word: Any,
    clue: Any,
    context: Any,
    domain_match: float,
) -> float:
    st_overlap = 0.5 * _char_overlap(source_word, target_word) + 0.5 * _token_overlap(source_word, target_word)
    tc_overlap = 0.5 * _char_overlap(target_word, clue) + 0.5 * _token_overlap(target_word, context)
    score = 1.0 - (0.45 * st_overlap + 0.25 * domain_match + 0.30 * tc_overlap)
    return float(max(0.0, min(1.0, score)))



def _compute_semantic_features(df: pd.DataFrame, cfg: SemanticFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    domain_match_col = f"{cfg.output_prefix}_usas_domain_match"
    ent_u_col = f"{cfg.output_prefix}_usas_entropy_unweighted"
    ent_w_col = f"{cfg.output_prefix}_usas_entropy_weighted"
    shift_col = f"{cfg.output_prefix}_semantic_shift"

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.compute_mode == "compute_if_missing":
        existing = [c for c in (domain_match_col, ent_u_col, ent_w_col, shift_col) if c in result.columns]
        if len(existing) == 4 and all(not result[c].isna().all() for c in existing):
            return result

    has_domain_cols = (
        cfg.source_domain_column in result.columns if cfg.source_domain_column else False
    ) and (
        cfg.target_domain_column in result.columns if cfg.target_domain_column else False
    )

    if not has_domain_cols and not cfg.fallback_when_domains_missing:
        raise FeatureValidationError(
            "USAS/domain columns are missing and semantic.fallback_when_domains_missing=false."
        )

    source_series = result[cfg.source_word_column] if cfg.source_word_column and cfg.source_word_column in result.columns else pd.Series([None] * len(result), index=result.index)
    target_series = result[cfg.target_word_column] if cfg.target_word_column and cfg.target_word_column in result.columns else pd.Series([None] * len(result), index=result.index)
    clue_series = result[cfg.clue_column] if cfg.clue_column and cfg.clue_column in result.columns else pd.Series([None] * len(result), index=result.index)
    context_series = result[cfg.context_column] if cfg.context_column and cfg.context_column in result.columns else pd.Series([None] * len(result), index=result.index)

    if has_domain_cols:
        src_dom_series = result[cfg.source_domain_column]
        tgt_dom_series = result[cfg.target_domain_column]
        src_score_series = result[cfg.source_domain_score_column] if cfg.source_domain_score_column and cfg.source_domain_score_column in result.columns else pd.Series([None] * len(result), index=result.index)
        tgt_score_series = result[cfg.target_domain_score_column] if cfg.target_domain_score_column and cfg.target_domain_score_column in result.columns else pd.Series([None] * len(result), index=result.index)
    else:
        src_dom_series = pd.Series([None] * len(result), index=result.index)
        tgt_dom_series = pd.Series([None] * len(result), index=result.index)
        src_score_series = pd.Series([None] * len(result), index=result.index)
        tgt_score_series = pd.Series([None] * len(result), index=result.index)

    domain_match_vals: list[float] = []
    ent_u_vals: list[float] = []
    ent_w_vals: list[float] = []
    shift_vals: list[float] = []

    for src_word, tgt_word, clue, context, src_dom, tgt_dom, src_scores, tgt_scores in zip(
        source_series,
        target_series,
        clue_series,
        context_series,
        src_dom_series,
        tgt_dom_series,
        src_score_series,
        tgt_score_series,
    ):
        src_domains = _parse_domain_list(src_dom)
        tgt_domains = _parse_domain_list(tgt_dom)

        domain_match = _domain_match_score(src_domains, tgt_domains)

        if src_domains or tgt_domains:
            combined_domains = src_domains + tgt_domains
            ent_u = _safe_entropy([1.0] * len(combined_domains), base=cfg.entropy_base)
            combined_scores = _parse_score_list(src_scores, len(src_domains)) + _parse_score_list(tgt_scores, len(tgt_domains))
            ent_w = _safe_entropy(combined_scores, base=cfg.entropy_base)
        else:
            # fallback when no domain annotations exist
            #ent_u = 1.0 - _token_overlap(source_word=src_word if False else src_word, b=tgt_word) if False else 0.0
            ent_u = 1.0 - _token_overlap(src_word, tgt_word)
            ent_w = 1.0 - _char_overlap(src_word, tgt_word)
            domain_match = 0.5 * _token_overlap(src_word, clue) if cfg.fallback_when_domains_missing else domain_match

        semantic_shift = _semantic_shift_score(
            source_word=src_word,
            target_word=tgt_word,
            clue=clue,
            context=context,
            domain_match=domain_match,
        )

        domain_match_vals.append(domain_match)
        ent_u_vals.append(ent_u)
        ent_w_vals.append(ent_w)
        shift_vals.append(semantic_shift)

    result[domain_match_col] = domain_match_vals
    result[ent_u_col] = ent_u_vals
    result[ent_w_col] = ent_w_vals
    result[shift_col] = shift_vals
    return result


# -------------------------------------------------
# Validation / postprocessing
# -------------------------------------------------
def _fill_missing_columns(df: pd.DataFrame, cfg: SemanticFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result



def _cast_columns(df: pd.DataFrame, cfg: SemanticFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result



def _clip_similarities(df: pd.DataFrame, cfg: SemanticFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    if not cfg.clip_similarity_to_unit_interval:
        return result

    for col in [
        f"{cfg.output_prefix}_usas_domain_match",
        f"{cfg.output_prefix}_semantic_shift",
    ]:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            result[col] = numeric.clip(lower=0.0, upper=1.0)
    return result



def _ensure_expected_columns(df: pd.DataFrame, cfg: SemanticFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Semantic feature group is missing expected columns in split '{split_name}': {missing}"
        )



def _validate_negative_constraints(df: pd.DataFrame, cfg: SemanticFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"Semantic column '{col}' contains negative values in split '{split_name}'."
            )



def _collect_feature_columns(df: pd.DataFrame, cfg: SemanticFeatureConfig) -> list[str]:
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
    Build USAS/domain-oriented semantic features.

    Main outputs
    ------------
    - semantic_usas_domain_match
    - semantic_usas_entropy_unweighted
    - semantic_usas_entropy_weighted
    - semantic_semantic_shift

    Design
    ------
    - If domain columns are available, compute domain overlap and entropy from them.
    - If domain columns are missing and fallback_when_domains_missing=true,
      compute conservative fallback proxies instead of failing.
    - This matches the planned UCREL/USAS-oriented design better than a generic
      surface-similarity placeholder.
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _compute_semantic_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)
    result = _clip_similarities(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    semantic_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["semantic_feature_columns"] = semantic_feature_columns
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
