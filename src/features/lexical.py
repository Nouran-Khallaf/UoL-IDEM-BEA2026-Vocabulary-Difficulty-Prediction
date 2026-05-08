from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import spacy

from src.core.exceptions import FeatureValidationError


@dataclass(slots=True)
class LexicalFeatureConfig:
    fillna: dict[str, Any]
    cast: dict[str, str]
    columns_expected: list[str]
    forbid_all_zero_columns: bool = False
    forbid_negative_values: list[str] | None = None


_DEFAULT_NUMERIC_FILL = {
    # -------------------------------------------------
    # Kelly-based learner-oriented lexical signals
    # -------------------------------------------------
    "kelly_rank": 0.0,
    "kelly_rank_percentile": 0.0,
    "kelly_points": 0.0,
    "kelly_found": 0,
    "kelly_cefr": "UNK",

    # -------------------------------------------------
    # wordfreq-based lexical frequency signals
    # -------------------------------------------------
    "wf_value": 0.0,
    "wf_zipf": 0.0,
    "wf_cost": 0.0,
    "wf_percentile": 0.0,
    "wf_found": 0,

    # -------------------------------------------------
    # SUBTLEX-based lexical frequency signals
    # -------------------------------------------------
    "subtlex_wf": 0.0,
    "subtlex_lg10wf": 0.0,
    "subtlex_cd": 0.0,
    "subtlex_lg10cd": 0.0,
    "subtlex_zipf": 0.0,
    "subtlex_found": 0,

    # -------------------------------------------------
    # Main lexical helpers
    # -------------------------------------------------
    "en_target_lemma": "",
    "target_len": 0,
    "source_len": 0,
    "target_syllables": 0,
    "source_syllables": 0,
    "clue_len": 0,
    "clue_visible_chars": 0,
    "clue_hidden_chars": 0,
    "clue_reveals_first_char": 0,
    "clue_matches_target_len": 0,
    "clue_hidden_target_len": 0,
    "pos_encoded": 0,
    "clue_ratio": 0.0,
    "target_in_source": 0,
    "source_in_target": 0,
    "target_prefix_in_source": 0,
    "target_suffix_in_source": 0,
    "shared_char_ratio": 0.0,

    # -------------------------------------------------
    # Compatibility / downstream alias columns
    # -------------------------------------------------
    "freq_en": 0.0,
    "freq_percentile": 0.0,
    "syllables": 0,
    "clue_length": 0,
    "clue_initial_match": 0,
    "target_vowel_ratio": 0.0,
    "target_consonant_ratio": 0.0,
    "source_char_len_norm": 0.0,
    "target_char_len_norm": 0.0,
}


_DEFAULT_CAST = {
    # Kelly
    "kelly_rank": "float",
    "kelly_rank_percentile": "float",
    "kelly_points": "float",
    "kelly_found": "int",
    "kelly_cefr": "string",

    # wordfreq
    "wf_value": "float",
    "wf_zipf": "float",
    "wf_cost": "float",
    "wf_percentile": "float",
    "wf_found": "int",

    # SUBTLEX
    "subtlex_wf": "float",
    "subtlex_lg10wf": "float",
    "subtlex_cd": "float",
    "subtlex_lg10cd": "float",
    "subtlex_zipf": "float",
    "subtlex_found": "int",

    # Main lexical helpers
    "en_target_lemma": "string",
    "target_len": "int",
    "source_len": "int",
    "target_syllables": "int",
    "source_syllables": "int",
    "clue_len": "int",
    "clue_visible_chars": "int",
    "clue_hidden_chars": "int",
    "clue_reveals_first_char": "int",
    "clue_matches_target_len": "int",
    "clue_hidden_target_len": "int",
    "pos_encoded": "int",
    "clue_ratio": "float",
    "target_in_source": "int",
    "source_in_target": "int",
    "target_prefix_in_source": "int",
    "target_suffix_in_source": "int",
    "shared_char_ratio": "float",

    # Alias columns
    "freq_en": "float",
    "freq_percentile": "float",
    "syllables": "int",
    "clue_length": "int",
    "clue_initial_match": "int",
    "target_vowel_ratio": "float",
    "target_consonant_ratio": "float",
    "source_char_len_norm": "float",
    "target_char_len_norm": "float",
}


_DEFAULT_EXPECTED = [
    # Kelly
    "kelly_rank",
    "kelly_rank_percentile",
    "kelly_points",
    "kelly_found",
    "kelly_cefr",

    # wordfreq
    "wf_value",
    "wf_zipf",
    "wf_cost",
    "wf_percentile",
    "wf_found",

    # SUBTLEX
    "subtlex_wf",
    "subtlex_lg10wf",
    "subtlex_cd",
    "subtlex_lg10cd",
    "subtlex_zipf",
    "subtlex_found",

    # Main lexical helpers
    "en_target_lemma",
    "target_len",
    "source_len",
    "target_syllables",
    "source_syllables",
    "clue_len",
    "clue_visible_chars",
    "clue_hidden_chars",
    "clue_reveals_first_char",
    "clue_matches_target_len",
    "clue_hidden_target_len",
    "pos_encoded",
    "clue_ratio",
    "target_in_source",
    "source_in_target",
    "target_prefix_in_source",
    "target_suffix_in_source",
    "shared_char_ratio",

    # Alias / compatibility features
    "freq_en",
    "freq_percentile",
    "syllables",
    "clue_length",
    "clue_initial_match",
    "target_vowel_ratio",
    "target_consonant_ratio",
    "source_char_len_norm",
    "target_char_len_norm",
]


_EN_NLP = None


def _get_en_nlp():
    global _EN_NLP
    if _EN_NLP is None:
        try:
            _EN_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        except Exception as e:
            raise FeatureValidationError(
                "Failed to load spaCy model 'en_core_web_sm' needed for en_target_lemma generation. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from e
    return _EN_NLP


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Lexical feature builder expects a pandas DataFrame.")
    return df


def _get_nested_dict(d: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    value = d.get(key, {})
    return value if isinstance(value, dict) else {}


def _parse_feature_group_cfg(feature_group_cfg: dict[str, Any] | None) -> LexicalFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for lexical features must be a dictionary.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("lexical.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("lexical.validation must be a dictionary.")

    fillna = dict(_DEFAULT_NUMERIC_FILL)
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = dict(_DEFAULT_CAST)
    cast.update(_get_nested_dict(preprocessing, "cast"))

    columns_expected = feature_group_cfg.get("columns_expected", _DEFAULT_EXPECTED)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("lexical.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    forbid_all_zero_columns = bool(validation.get("forbid_all_zero_columns", False))
    forbid_negative_values = validation.get("forbid_negative_values", [])
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("lexical.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    return LexicalFeatureConfig(
        fillna=fillna,
        cast=cast,
        columns_expected=columns_expected,
        forbid_all_zero_columns=forbid_all_zero_columns,
        forbid_negative_values=forbid_negative_values,
    )


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception as e:
        raise FeatureValidationError(
            f"Failed to coerce lexical column '{column_name}' to numeric: {e}"
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
        f"Unsupported lexical cast type '{target_type}' for column '{column_name}'."
    )


def _fill_missing_lexical_columns(df: pd.DataFrame, cfg: LexicalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result


def _cast_lexical_columns(df: pd.DataFrame, cfg: LexicalFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result


def _ensure_expected_columns(df: pd.DataFrame, cfg: LexicalFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Lexical feature group is missing expected columns in split '{split_name}': {missing}"
        )


def _validate_negative_constraints(df: pd.DataFrame, cfg: LexicalFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values or []:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"Lexical column '{col}' contains negative values in split '{split_name}'."
            )


def _validate_all_zero_constraints(df: pd.DataFrame, cfg: LexicalFeatureConfig, split_name: str) -> None:
    if not cfg.forbid_all_zero_columns:
        return

    offending: list[str] = []
    for col in cfg.columns_expected:
        if col not in df.columns:
            continue

        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if len(numeric) > 0 and float(np.abs(numeric).sum()) == 0.0:
            offending.append(col)

    if offending:
        raise FeatureValidationError(
            f"Lexical feature group produced all-zero columns in split '{split_name}': {offending}"
        )


def _safe_text_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate].astype("string")
    return None


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def _letters_only(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z]", "", str(text).lower())


def _count_syllables_simple(word: str) -> int:
    if not word:
        return 0

    word = _letters_only(word)
    if not word:
        return 0

    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False

    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    if word.endswith("e") and not word.endswith(("le", "ye")) and syllables > 1:
        syllables -= 1

    return max(1, syllables)


def _shared_char_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ca = Counter(a)
    cb = Counter(b)
    overlap = sum((ca & cb).values())
    denom = max(len(a), len(b), 1)
    return float(overlap) / float(denom)


def _char_type_ratios(word: str) -> tuple[float, float]:
    cleaned = _letters_only(word)
    if not cleaned:
        return 0.0, 0.0

    vowels = sum(1 for ch in cleaned if ch in "aeiou")
    consonants = sum(1 for ch in cleaned if ch.isalpha() and ch not in "aeiou")
    denom = max(len(cleaned), 1)
    return float(vowels) / float(denom), float(consonants) / float(denom)


def _lemmatize_english_targets(series: pd.Series) -> pd.Series:
    nlp = _get_en_nlp()
    texts = series.fillna("").astype(str).tolist()
    lemmas: list[str] = []

    for doc in nlp.pipe(texts, batch_size=256):
        tokens = [t for t in doc if not t.is_space]
        if not tokens:
            lemmas.append("")
            continue

        if len(tokens) == 1:
            lemma = tokens[0].lemma_.strip().lower()
            if not lemma:
                lemma = tokens[0].text.strip().lower()
            lemmas.append(lemma)
            continue

        lemma_parts: list[str] = []
        for tok in tokens:
            lemma = tok.lemma_.strip().lower()
            lemma_parts.append(lemma if lemma else tok.text.strip().lower())
        lemmas.append(" ".join(lemma_parts).strip())

    return pd.Series(lemmas, index=series.index, dtype="string")


def _derive_en_target_lemma(result: pd.DataFrame) -> pd.DataFrame:
    if "en_target_lemma" in result.columns:
        existing = result["en_target_lemma"].astype("string").fillna("").str.strip()
        if (existing != "").any():
            return result

    target_word_series = _safe_text_series(
        result,
        ("en_target_word", "en_word", "target_word"),
    )
    if target_word_series is None:
        return result

    result["en_target_lemma"] = _lemmatize_english_targets(target_word_series)
    return result


def _derive_optional_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add safe derived lexical helpers only when they are missing.
    Also populate compatibility alias columns that some configs still expect.
    """
    result = df.copy()

    # First: derive lemma
    result = _derive_en_target_lemma(result)

    target_series = _safe_text_series(
        result,
        ("en_target_lemma", "en_target_word", "en_word", "target_word"),
    )
    source_series = _safe_text_series(
        result,
        ("l1_source_lemma", "L1_source_word", "source_word"),
    )
    clue_series = _safe_text_series(
        result,
        ("en_target_clue", "target_clue", "clue"),
    )

    target_norm = _normalized_text(target_series) if target_series is not None else None
    source_norm = _normalized_text(source_series) if source_series is not None else None
    clue_raw = clue_series.fillna("").astype("string").str.strip() if clue_series is not None else None

    # -------------------------------------------------
    # Target-derived features
    # -------------------------------------------------
    if target_norm is not None:
        if "target_len" not in result.columns:
            result["target_len"] = target_norm.str.len().fillna(0).astype(int)

        if "target_syllables" not in result.columns:
            result["target_syllables"] = target_norm.map(_count_syllables_simple).astype(int)

    # -------------------------------------------------
    # Source-derived features
    # -------------------------------------------------
    if source_norm is not None:
        if "source_len" not in result.columns:
            result["source_len"] = source_norm.str.len().fillna(0).astype(int)

        if "source_syllables" not in result.columns:
            result["source_syllables"] = source_norm.map(_count_syllables_simple).astype(int)

    # -------------------------------------------------
    # Clue-derived features
    # -------------------------------------------------
    if clue_raw is not None:
        if "clue_len" not in result.columns:
            result["clue_len"] = clue_raw.str.len().fillna(0).astype(int)

        if "clue_visible_chars" not in result.columns:
            result["clue_visible_chars"] = (
                clue_raw.str.replace("_", "", regex=False).str.len().fillna(0).astype(int)
            )

        if "clue_hidden_chars" not in result.columns:
            result["clue_hidden_chars"] = clue_raw.str.count("_").fillna(0).astype(int)

        if "clue_reveals_first_char" not in result.columns:
            result["clue_reveals_first_char"] = clue_raw.str.match(r"^[^\W_]", na=False).astype(int)

        if target_norm is not None:
            target_len = target_norm.str.len().fillna(0).astype(int)
            visible_chars = clue_raw.str.replace("_", "", regex=False).str.len().fillna(0).astype(int)
            clue_len = clue_raw.str.len().fillna(0).astype(int)

            if "clue_matches_target_len" not in result.columns:
                result["clue_matches_target_len"] = (clue_len == target_len).astype(int)

            if "clue_ratio" not in result.columns:
                safe_target_len = target_len.replace(0, np.nan)
                result["clue_ratio"] = (visible_chars / safe_target_len).fillna(0.0).astype(float)

            if "clue_hidden_target_len" not in result.columns:
                result["clue_hidden_target_len"] = (target_len - visible_chars).clip(lower=0).astype(int)

    # -------------------------------------------------
    # Target/source overlap features
    # -------------------------------------------------
    if target_norm is not None and source_norm is not None:
        target_values = target_norm.tolist()
        source_values = source_norm.tolist()

        if "target_in_source" not in result.columns:
            result["target_in_source"] = [
                int(bool(t) and bool(s) and (t in s))
                for t, s in zip(target_values, source_values)
            ]

        if "source_in_target" not in result.columns:
            result["source_in_target"] = [
                int(bool(t) and bool(s) and (s in t))
                for t, s in zip(target_values, source_values)
            ]

        if "target_prefix_in_source" not in result.columns:
            result["target_prefix_in_source"] = [
                int(bool(t) and len(t) >= 3 and (t[:3] in s))
                for t, s in zip(target_values, source_values)
            ]

        if "target_suffix_in_source" not in result.columns:
            result["target_suffix_in_source"] = [
                int(bool(t) and len(t) >= 3 and (t[-3:] in s))
                for t, s in zip(target_values, source_values)
            ]

        if "shared_char_ratio" not in result.columns:
            result["shared_char_ratio"] = [
                _shared_char_ratio(t, s)
                for t, s in zip(target_values, source_values)
            ]

    # -------------------------------------------------
    # Compatibility / alias features
    # -------------------------------------------------
    if "freq_en" not in result.columns:
        if "wf_zipf" in result.columns:
            result["freq_en"] = pd.to_numeric(result["wf_zipf"], errors="coerce").fillna(0.0).astype(float)
        elif "wf_value" in result.columns:
            result["freq_en"] = pd.to_numeric(result["wf_value"], errors="coerce").fillna(0.0).astype(float)
        elif "subtlex_zipf" in result.columns:
            result["freq_en"] = pd.to_numeric(result["subtlex_zipf"], errors="coerce").fillna(0.0).astype(float)

    if "freq_percentile" not in result.columns:
        if "wf_percentile" in result.columns:
            result["freq_percentile"] = (
                pd.to_numeric(result["wf_percentile"], errors="coerce").fillna(0.0).astype(float)
            )
        elif "kelly_rank_percentile" in result.columns:
            result["freq_percentile"] = (
                pd.to_numeric(result["kelly_rank_percentile"], errors="coerce").fillna(0.0).astype(float)
            )

    if "syllables" not in result.columns and "target_syllables" in result.columns:
        result["syllables"] = pd.to_numeric(result["target_syllables"], errors="coerce").fillna(0).astype(int)

    if "clue_length" not in result.columns and "clue_len" in result.columns:
        result["clue_length"] = pd.to_numeric(result["clue_len"], errors="coerce").fillna(0).astype(int)

    if "clue_initial_match" not in result.columns:
        if target_norm is not None and clue_raw is not None:
            clue_first = clue_raw.str.extract(r"([A-Za-z])", expand=False).fillna("").str.lower()
            target_first = target_norm.str[:1].fillna("")
            result["clue_initial_match"] = (
                (clue_first != "") & (target_first != "") & (clue_first == target_first)
            ).astype(int)

    if "target_vowel_ratio" not in result.columns and target_norm is not None:
        ratios = target_norm.map(_char_type_ratios)
        result["target_vowel_ratio"] = ratios.map(lambda x: x[0]).astype(float)

    if "target_consonant_ratio" not in result.columns and target_norm is not None:
        ratios = target_norm.map(_char_type_ratios)
        result["target_consonant_ratio"] = ratios.map(lambda x: x[1]).astype(float)

    if "source_char_len_norm" not in result.columns and "source_len" in result.columns:
        src_len = pd.to_numeric(result["source_len"], errors="coerce").fillna(0.0)
        src_max = float(src_len.max()) if len(src_len) > 0 else 0.0
        denom = src_max if src_max > 0 else 1.0
        result["source_char_len_norm"] = (src_len / denom).astype(float)

    if "target_char_len_norm" not in result.columns and "target_len" in result.columns:
        tgt_len = pd.to_numeric(result["target_len"], errors="coerce").fillna(0.0)
        tgt_max = float(tgt_len.max()) if len(tgt_len) > 0 else 0.0
        denom = tgt_max if tgt_max > 0 else 1.0
        result["target_char_len_norm"] = (tgt_len / denom).astype(float)

    def _replace_if_all_zero_or_missing(dst: str, src: str) -> None:
        if dst not in result.columns or src not in result.columns:
            return
        dst_num = pd.to_numeric(result[dst], errors="coerce").fillna(0)
        src_num = pd.to_numeric(result[src], errors="coerce").fillna(0)
        if float(np.abs(dst_num).sum()) == 0.0 and float(np.abs(src_num).sum()) > 0.0:
            result[dst] = src_num

    _replace_if_all_zero_or_missing("freq_en", "wf_zipf")
    _replace_if_all_zero_or_missing("freq_percentile", "wf_percentile")
    _replace_if_all_zero_or_missing("syllables", "target_syllables")
    _replace_if_all_zero_or_missing("clue_length", "clue_len")

    return result


def _collect_feature_columns(df: pd.DataFrame, cfg: LexicalFeatureConfig) -> list[str]:
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
    Build/validate lexical features for one split.

    Notes:
    - Kelly / wordfreq / SUBTLEX values are expected to be computed upstream.
    - This module standardizes, fills, casts, validates them.
    - It also derives en_target_lemma using spaCy from en_target_word.
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg)

    result = df.copy()
    result = _derive_optional_lexical_features(result)
    result = _fill_missing_lexical_columns(result, parsed)
    result = _cast_lexical_columns(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)
    _validate_all_zero_constraints(result, parsed, split_name)

    lexical_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["lexical_feature_columns"] = lexical_feature_columns
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