from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import FeatureValidationError


SUPPORTED_LANGS_FOR_POS = {
    "en": {"spacy_core": "en_core_web_sm"},
    "es": {"spacy_core": "es_core_news_sm"},
    "cmn": {"spacy_core": "zh_core_web_sm"},
    "zh": {"spacy_core": "zh_core_web_sm"},
    "de": {"spacy_core": "nl_core_news_sm"},
    "spa": {"spacy_core": "es_core_news_sm"},
    "zho": {"spacy_core": "zh_core_web_sm"},
    "ger": {"spacy_core": "de_core_news_sm"},
    "deu": {"spacy_core": "de_core_news_sm"},
}


@dataclass(slots=True)
class FrequencyFeatureConfig:
    l1_column: str
    source_word_column: str
    target_word_column: str
    target_lemma_column: str | None
    target_pos_column: str | None

    kelly_enabled: bool
    kelly_path: str | None
    kelly_sheet_name: str | int | None
    kelly_word_column: str
    kelly_pos_column: str | None
    kelly_rank_column: str | None
    kelly_cefr_column: str | None
    kelly_points_column: str | None

    wordfreq_enabled: bool
    wordfreq_language: str
    wordfreq_epsilon: float

    subtlex_enabled: bool
    subtlex_path: str | None
    subtlex_sheet_name: str | int | None
    subtlex_word_column: str
    subtlex_pos_column: str | None
    subtlex_wf_column: str | None
    subtlex_lg10wf_column: str | None
    subtlex_cd_column: str | None
    subtlex_lg10cd_column: str | None
    subtlex_zipf_column: str | None

    lowercase_lookup: bool
    strip_lookup: bool
    replace_nan_strings: bool
    pos_map: dict[str, str]
    pos_code_map: dict[str, int]


_DEFAULTS: dict[str, Any] = {
    "l1_column": "L1",
    "source_word_column": "L1_source_word",
    "target_word_column": "en_target_word",
    "target_lemma_column": "en_target_lemma",
    "target_pos_column": "en_target_pos",

    "kelly_enabled": True,
    "kelly_path": None,
    "kelly_sheet_name": 0,
    "kelly_word_column": "Word",
    "kelly_pos_column": "Part of Speech",
    "kelly_rank_column": "ID number",
    "kelly_cefr_column": "CEFR",
    "kelly_points_column": "Points",

    "wordfreq_enabled": True,
    "wordfreq_language": "en",
    "wordfreq_epsilon": 1e-9,

    "subtlex_enabled": False,
    "subtlex_path": None,
    "subtlex_sheet_name": 0,
    "subtlex_word_column": "Word",
    "subtlex_pos_column": None,
    "subtlex_wf_column": "SUBTLWF",
    "subtlex_lg10wf_column": "Lg10WF",
    "subtlex_cd_column": "SUBTLCD",
    "subtlex_lg10cd_column": "Lg10CD",
    "subtlex_zipf_column": "Zipf",

    "lowercase_lookup": True,
    "strip_lookup": True,
    "replace_nan_strings": True,

    "pos_map": {
        "noun": "NOUN",
        "n": "NOUN",
        "nn": "NOUN",
        "propn": "PROPN",
        "proper noun": "PROPN",

        "verb": "VERB",
        "v": "VERB",
        "vv": "VERB",

        "adj": "ADJ",
        "adjective": "ADJ",
        "jj": "ADJ",

        "adv": "ADV",
        "adverb": "ADV",
        "rb": "ADV",

        "pron": "PRON",
        "pronoun": "PRON",

        "det": "DET",
        "determiner": "DET",

        "adp": "ADP",
        "prep": "ADP",
        "preposition": "ADP",

        "conj": "CCONJ",
        "cconj": "CCONJ",
        "coord conj": "CCONJ",
        "coordinating conjunction": "CCONJ",
        "sconj": "SCONJ",
        "subordinating conjunction": "SCONJ",

        "num": "NUM",
        "number": "NUM",

        "intj": "INTJ",
        "interjection": "INTJ",

        "aux": "AUX",
        "part": "PART",
        "particle": "PART",
    },

    "pos_code_map": {
        "": 0,
        "UNK": 0,
        "NOUN": 1,
        "PROPN": 2,
        "VERB": 3,
        "AUX": 4,
        "ADJ": 5,
        "ADV": 6,
        "PRON": 7,
        "DET": 8,
        "ADP": 9,
        "CCONJ": 10,
        "SCONJ": 11,
        "NUM": 12,
        "PART": 13,
        "INTJ": 14,
    },
}


_SPACY_MODEL_CACHE: dict[str, Any] = {}


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Frequency feature builder expects a pandas DataFrame.")
    return df


def _parse_feature_group_cfg(feature_group_cfg: dict[str, Any] | None) -> FrequencyFeatureConfig:
    raw = dict(_DEFAULTS)
    if feature_group_cfg is not None:
        if not isinstance(feature_group_cfg, dict):
            raise FeatureValidationError("frequency feature_group_cfg must be a dictionary.")
        raw.update(feature_group_cfg)

    pos_map = raw.get("pos_map", {})
    if not isinstance(pos_map, dict):
        raise FeatureValidationError("frequency.pos_map must be a dictionary.")

    pos_code_map = raw.get("pos_code_map", {})
    if not isinstance(pos_code_map, dict):
        raise FeatureValidationError("frequency.pos_code_map must be a dictionary.")

    return FrequencyFeatureConfig(
        l1_column=str(raw["l1_column"]),
        source_word_column=str(raw["source_word_column"]),
        target_word_column=str(raw["target_word_column"]),
        target_lemma_column=None if raw.get("target_lemma_column") in (None, "") else str(raw["target_lemma_column"]),
        target_pos_column=None if raw.get("target_pos_column") in (None, "") else str(raw["target_pos_column"]),

        kelly_enabled=bool(raw["kelly_enabled"]),
        kelly_path=None if raw.get("kelly_path") in (None, "") else str(raw["kelly_path"]),
        kelly_sheet_name=raw.get("kelly_sheet_name", 0),
        kelly_word_column=str(raw["kelly_word_column"]),
        kelly_pos_column=None if raw.get("kelly_pos_column") in (None, "") else str(raw["kelly_pos_column"]),
        kelly_rank_column=None if raw.get("kelly_rank_column") in (None, "") else str(raw["kelly_rank_column"]),
        kelly_cefr_column=None if raw.get("kelly_cefr_column") in (None, "") else str(raw["kelly_cefr_column"]),
        kelly_points_column=None if raw.get("kelly_points_column") in (None, "") else str(raw["kelly_points_column"]),

        wordfreq_enabled=bool(raw["wordfreq_enabled"]),
        wordfreq_language=str(raw["wordfreq_language"]),
        wordfreq_epsilon=float(raw["wordfreq_epsilon"]),

        subtlex_enabled=bool(raw["subtlex_enabled"]),
        subtlex_path=None if raw.get("subtlex_path") in (None, "") else str(raw["subtlex_path"]),
        subtlex_sheet_name=raw.get("subtlex_sheet_name", 0),
        subtlex_word_column=str(raw["subtlex_word_column"]),
        subtlex_pos_column=None if raw.get("subtlex_pos_column") in (None, "") else str(raw["subtlex_pos_column"]),
        subtlex_wf_column=None if raw.get("subtlex_wf_column") in (None, "") else str(raw["subtlex_wf_column"]),
        subtlex_lg10wf_column=None if raw.get("subtlex_lg10wf_column") in (None, "") else str(raw["subtlex_lg10wf_column"]),
        subtlex_cd_column=None if raw.get("subtlex_cd_column") in (None, "") else str(raw["subtlex_cd_column"]),
        subtlex_lg10cd_column=None if raw.get("subtlex_lg10cd_column") in (None, "") else str(raw["subtlex_lg10cd_column"]),
        subtlex_zipf_column=None if raw.get("subtlex_zipf_column") in (None, "") else str(raw["subtlex_zipf_column"]),

        lowercase_lookup=bool(raw["lowercase_lookup"]),
        strip_lookup=bool(raw["strip_lookup"]),
        replace_nan_strings=bool(raw["replace_nan_strings"]),
        pos_map={str(k): str(v) for k, v in pos_map.items()},
        pos_code_map={str(k): int(v) for k, v in pos_code_map.items()},
    )


def _normalize_text_value(
    value: Any,
    *,
    lowercase: bool,
    strip: bool,
    replace_nan_strings: bool,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    if replace_nan_strings and text.strip().lower() in {"nan", "none", "<na>"}:
        return ""

    if strip:
        text = text.strip()
    if lowercase:
        text = text.lower()

    return text


def _normalize_series(
    series: pd.Series,
    *,
    lowercase: bool,
    strip: bool,
    replace_nan_strings: bool,
) -> pd.Series:
    return series.map(
        lambda x: _normalize_text_value(
            x,
            lowercase=lowercase,
            strip=strip,
            replace_nan_strings=replace_nan_strings,
        )
    )


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_pos_value(value: Any, pos_map: dict[str, str]) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in pos_map:
        return pos_map[lowered]
    return text.upper()


def _normalize_pos_series(series: pd.Series, pos_map: dict[str, str]) -> pd.Series:
    return series.map(lambda x: _normalize_pos_value(x, pos_map))


def _encode_pos_series(series: pd.Series, pos_code_map: dict[str, int]) -> pd.Series:
    return (
        series.fillna("")
        .astype("string")
        .map(lambda x: pos_code_map.get(str(x), 0))
        .fillna(0)
        .astype(int)
    )


def _compute_percentile_from_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = numeric.notna()
    if mask.any():
        out.loc[mask] = numeric.loc[mask].rank(method="average", pct=True).astype(float)
    return out


def _compute_percentile_from_rank(rank_values: pd.Series, ascending_rank_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(rank_values, errors="coerce")
    out = pd.Series(np.nan, index=rank_values.index, dtype=float)

    mask = numeric.notna()
    if not mask.any():
        return out

    valid = numeric.loc[mask].astype(float)

    if ascending_rank_is_better:
        out.loc[mask] = (-valid).rank(method="average", pct=True).astype(float)
    else:
        out.loc[mask] = valid.rank(method="average", pct=True).astype(float)

    return out


def _resolve_lookup_word_series(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.Series:
    if cfg.target_word_column not in df.columns:
        raise FeatureValidationError(
            f"Frequency features require target word column '{cfg.target_word_column}'."
        )

    word = _normalize_series(
        df[cfg.target_word_column],
        lowercase=cfg.lowercase_lookup,
        strip=cfg.strip_lookup,
        replace_nan_strings=cfg.replace_nan_strings,
    )

    if cfg.target_lemma_column and cfg.target_lemma_column in df.columns:
        lemma = _normalize_series(
            df[cfg.target_lemma_column],
            lowercase=cfg.lowercase_lookup,
            strip=cfg.strip_lookup,
            replace_nan_strings=cfg.replace_nan_strings,
        )
        return lemma.where(lemma != "", word)

    return word


def _resolve_target_pos_series(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.Series:
    if cfg.target_pos_column and cfg.target_pos_column in df.columns:
        return _normalize_pos_series(df[cfg.target_pos_column], cfg.pos_map)
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _resolve_spacy_model_name(l1_value: str) -> str | None:
    key = str(l1_value).strip().lower()
    info = SUPPORTED_LANGS_FOR_POS.get(key)
    if info is None:
        return None
    return info["spacy_core"]


def _get_spacy_model(model_name: str):
    if model_name in _SPACY_MODEL_CACHE:
        return _SPACY_MODEL_CACHE[model_name]

    try:
        import spacy
        nlp = spacy.load(model_name, disable=["ner", "parser", "lemmatizer", "textcat"])
    except Exception as e:
        raise FeatureValidationError(
            f"Could not load spaCy model '{model_name}'. "
            f"Install it first, e.g. python -m spacy download {model_name}"
        ) from e

    _SPACY_MODEL_CACHE[model_name] = nlp
    return nlp


def _infer_source_pos_for_row(source_word: Any, l1_value: Any, cfg: FrequencyFeatureConfig) -> str:
    if pd.isna(source_word):
        return ""

    text = str(source_word).strip()
    if not text:
        return ""

    model_name = _resolve_spacy_model_name(str(l1_value))
    if model_name is None:
        return ""

    nlp = _get_spacy_model(model_name)
    doc = nlp(text)

    for tok in doc:
        if not tok.is_space and not tok.is_punct:
            return _normalize_pos_value(tok.pos_, cfg.pos_map)

    return ""


def _build_source_pos_series(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.Series:
    if cfg.source_word_column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    if cfg.l1_column not in df.columns:
        raise FeatureValidationError(
            f"Frequency features require language column '{cfg.l1_column}' to generate source POS."
        )

    values = [
        _infer_source_pos_for_row(source_word, l1_value, cfg)
        for source_word, l1_value in zip(df[cfg.source_word_column].tolist(), df[cfg.l1_column].tolist())
    ]
    return pd.Series(values, index=df.index, dtype="string")


def _add_pos_features(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    target_pos = _resolve_target_pos_series(result, cfg)
    source_pos = _build_source_pos_series(result, cfg)

    result["target_pos_norm"] = target_pos.astype("string")
    result["source_pos_norm"] = source_pos.astype("string")

    result["target_pos_code"] = _encode_pos_series(result["target_pos_norm"], cfg.pos_code_map)
    result["source_pos_code"] = _encode_pos_series(result["source_pos_norm"], cfg.pos_code_map)

    result["target_pos_known"] = (result["target_pos_norm"] != "").astype(int)
    result["source_pos_known"] = (result["source_pos_norm"] != "").astype(int)

    result["source_target_pos_match"] = (
        (result["target_pos_norm"] != "")
        & (result["source_pos_norm"] != "")
        & (result["target_pos_norm"] == result["source_pos_norm"])
    ).astype(int)

    return result


def _load_table(path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FeatureValidationError(f"Resource file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep="\t")
    raise FeatureValidationError(f"Unsupported resource file format: {file_path}")


def _make_pair_key(word: str, pos: str) -> str:
    return f"{word}|||{pos}"


def _lookup_with_pos_fallback(
    lookup_word: pd.Series,
    lookup_pos: pd.Series,
    pair_to_value: dict[str, Any],
    word_to_value: dict[str, Any],
) -> pd.Series:
    values: list[Any] = []
    for word, pos in zip(lookup_word.tolist(), lookup_pos.tolist()):
        if word:
            pair_key = _make_pair_key(word, pos)
            if pos and pair_key in pair_to_value:
                values.append(pair_to_value[pair_key])
                continue
            if word in word_to_value:
                values.append(word_to_value[word])
                continue
        values.append(np.nan)
    return pd.Series(values, index=lookup_word.index)


def _lookup_found_with_pos_fallback(
    lookup_word: pd.Series,
    lookup_pos: pd.Series,
    pair_keys: set[str],
    word_keys: set[str],
) -> pd.Series:
    found: list[int] = []
    for word, pos in zip(lookup_word.tolist(), lookup_pos.tolist()):
        if not word:
            found.append(0)
            continue
        pair_key = _make_pair_key(word, pos)
        if pos and pair_key in pair_keys:
            found.append(1)
        elif word in word_keys:
            found.append(1)
        else:
            found.append(0)
    return pd.Series(found, index=lookup_word.index, dtype=int)


def _lookup_pos_exact_match(
    lookup_word: pd.Series,
    lookup_pos: pd.Series,
    pair_keys: set[str],
) -> pd.Series:
    matched: list[int] = []
    for word, pos in zip(lookup_word.tolist(), lookup_pos.tolist()):
        if not word or not pos:
            matched.append(0)
            continue
        matched.append(int(_make_pair_key(word, pos) in pair_keys))
    return pd.Series(matched, index=lookup_word.index, dtype=int)


def _build_resource_lookup_maps(
    resource_df: pd.DataFrame,
    *,
    word_column: str,
    pos_column: str | None,
    value_columns: dict[str, str],
    cfg: FrequencyFeatureConfig,
    dedupe_score_column: str | None = None,
    dedupe_score_ascending: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame]:
    df = resource_df.copy()

    if word_column not in df.columns:
        raise FeatureValidationError(f"Required resource word column '{word_column}' not found.")

    df["_lookup_word"] = _normalize_series(
        df[word_column],
        lowercase=cfg.lowercase_lookup,
        strip=cfg.strip_lookup,
        replace_nan_strings=cfg.replace_nan_strings,
    )

    if pos_column and pos_column in df.columns:
        df["_lookup_pos"] = _normalize_pos_series(df[pos_column], cfg.pos_map)
    else:
        df["_lookup_pos"] = ""

    df = df[df["_lookup_word"] != ""].copy()

    prepared_value_columns: dict[str, str] = {}
    for out_name, src_col in value_columns.items():
        if src_col in df.columns:
            prepared_col = f"__{out_name}"
            if out_name.endswith("_cefr"):
                df[prepared_col] = (
                    df[src_col].astype("string").fillna("UNK").replace({"": "UNK"})
                )
            else:
                df[prepared_col] = _safe_numeric(df[src_col])
            prepared_value_columns[out_name] = prepared_col

    if dedupe_score_column and dedupe_score_column in prepared_value_columns:
        score_col = prepared_value_columns[dedupe_score_column]
        df = df.sort_values(
            by=["_lookup_word", "_lookup_pos", score_col],
            ascending=[True, True, dedupe_score_ascending],
            na_position="last",
        )
    else:
        df = df.sort_values(by=["_lookup_word", "_lookup_pos"], ascending=[True, True])

    df_pair = df.drop_duplicates(subset=["_lookup_word", "_lookup_pos"], keep="first").copy()

    if dedupe_score_column and dedupe_score_column in prepared_value_columns:
        score_col = prepared_value_columns[dedupe_score_column]
        df_word = (
            df.sort_values(
                by=["_lookup_word", score_col],
                ascending=[True, dedupe_score_ascending],
                na_position="last",
            )
            .drop_duplicates(subset=["_lookup_word"], keep="first")
            .copy()
        )
    else:
        df_word = df.drop_duplicates(subset=["_lookup_word"], keep="first").copy()

    pair_maps: dict[str, dict[str, Any]] = {}
    word_maps: dict[str, dict[str, Any]] = {}

    for out_name, prepared_col in prepared_value_columns.items():
        pair_maps[out_name] = {
            _make_pair_key(word, pos): value
            for word, pos, value in zip(
                df_pair["_lookup_word"].tolist(),
                df_pair["_lookup_pos"].tolist(),
                df_pair[prepared_col].tolist(),
            )
        }
        word_maps[out_name] = {
            word: value
            for word, value in zip(
                df_word["_lookup_word"].tolist(),
                df_word[prepared_col].tolist(),
            )
        }

    return pair_maps, word_maps, df


def _build_kelly_features(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    result["kelly_rank"] = np.nan
    result["kelly_rank_percentile"] = np.nan
    result["kelly_points"] = np.nan
    result["kelly_cefr"] = pd.Series(["UNK"] * len(result), index=result.index, dtype="string")
    result["kelly_found"] = 0
    result["kelly_pos_match"] = 0

    if not cfg.kelly_enabled:
        return result
    if not cfg.kelly_path:
        raise FeatureValidationError("kelly_enabled=True but no kelly_path was provided.")

    lookup_word = _resolve_lookup_word_series(result, cfg)
    lookup_pos = _resolve_target_pos_series(result, cfg)

    kelly_df = _load_table(cfg.kelly_path, cfg.kelly_sheet_name)

    value_columns: dict[str, str] = {}
    if cfg.kelly_rank_column and cfg.kelly_rank_column in kelly_df.columns:
        value_columns["kelly_rank"] = cfg.kelly_rank_column
    if cfg.kelly_points_column and cfg.kelly_points_column in kelly_df.columns:
        value_columns["kelly_points"] = cfg.kelly_points_column
    if cfg.kelly_cefr_column and cfg.kelly_cefr_column in kelly_df.columns:
        value_columns["kelly_cefr"] = cfg.kelly_cefr_column

    pair_maps, word_maps, prepared_df = _build_resource_lookup_maps(
        kelly_df,
        word_column=cfg.kelly_word_column,
        pos_column=cfg.kelly_pos_column,
        value_columns=value_columns,
        cfg=cfg,
        dedupe_score_column="kelly_rank",
        dedupe_score_ascending=True,
    )

    if "kelly_rank" in pair_maps:
        result["kelly_rank"] = _lookup_with_pos_fallback(
            lookup_word, lookup_pos, pair_maps["kelly_rank"], word_maps["kelly_rank"]
        ).astype(float)

    if "kelly_points" in pair_maps:
        result["kelly_points"] = _lookup_with_pos_fallback(
            lookup_word, lookup_pos, pair_maps["kelly_points"], word_maps["kelly_points"]
        ).astype(float)

    if "kelly_cefr" in pair_maps:
        result["kelly_cefr"] = _lookup_with_pos_fallback(
            lookup_word, lookup_pos, pair_maps["kelly_cefr"], word_maps["kelly_cefr"]
        ).fillna("UNK").astype("string")

    pair_keys = set()
    word_keys = set()
    if not prepared_df.empty:
        pair_keys = {
            _make_pair_key(word, pos)
            for word, pos in zip(prepared_df["_lookup_word"], prepared_df["_lookup_pos"])
        }
        word_keys = set(prepared_df["_lookup_word"].tolist())

    result["kelly_found"] = _lookup_found_with_pos_fallback(
        lookup_word, lookup_pos, pair_keys, word_keys
    )
    result["kelly_pos_match"] = _lookup_pos_exact_match(
        lookup_word, lookup_pos, pair_keys
    )

    if "kelly_rank" in pair_maps and result["kelly_rank"].notna().any():
        result["kelly_rank_percentile"] = _compute_percentile_from_rank(
            result["kelly_rank"], ascending_rank_is_better=True
        ).fillna(0.0).astype(float)

    return result


def _build_wordfreq_features(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    result["wf_value"] = np.nan
    result["wf_zipf"] = np.nan
    result["wf_cost"] = np.nan
    result["wf_percentile"] = np.nan
    result["wf_found"] = 0

    if not cfg.wordfreq_enabled:
        return result

    try:
        from wordfreq import word_frequency, zipf_frequency
    except Exception as e:
        raise FeatureValidationError(
            "wordfreq is enabled but the package is not installed. Install it with: pip install wordfreq"
        ) from e

    lookup_word = _resolve_lookup_word_series(result, cfg)
    unique_words = pd.Index(lookup_word.dropna().unique())
    unique_words = unique_words[unique_words != ""]

    wf_map: dict[str, float] = {}
    zipf_map: dict[str, float] = {}

    for word in unique_words:
        try:
            wf_val = float(word_frequency(word, cfg.wordfreq_language))
        except Exception:
            wf_val = 0.0

        try:
            zipf_val = float(zipf_frequency(word, cfg.wordfreq_language))
        except Exception:
            zipf_val = np.nan

        wf_map[word] = wf_val
        zipf_map[word] = zipf_val

    result["wf_value"] = lookup_word.map(wf_map).fillna(0.0).astype(float)
    result["wf_zipf"] = lookup_word.map(zipf_map).astype(float)
    result["wf_found"] = (result["wf_value"] > 0).astype(int)

    eps = float(cfg.wordfreq_epsilon)
    if eps <= 0:
        raise FeatureValidationError("wordfreq_epsilon must be > 0.")

    safe_freq = result["wf_value"].clip(lower=eps)
    result["wf_cost"] = (-np.log(safe_freq)).astype(float)
    result["wf_percentile"] = _compute_percentile_from_values(result["wf_value"]).fillna(0.0).astype(float)

    return result


def _build_subtlex_features(df: pd.DataFrame, cfg: FrequencyFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    result["subtlex_wf"] = np.nan
    result["subtlex_lg10wf"] = np.nan
    result["subtlex_cd"] = np.nan
    result["subtlex_lg10cd"] = np.nan
    result["subtlex_zipf"] = np.nan
    result["subtlex_found"] = 0
    result["subtlex_pos_match"] = 0

    if not cfg.subtlex_enabled:
        return result
    if not cfg.subtlex_path:
        raise FeatureValidationError("subtlex_enabled=True but no subtlex_path was provided.")

    lookup_word = _resolve_lookup_word_series(result, cfg)
    lookup_pos = _resolve_target_pos_series(result, cfg)

    subtlex_df = _load_table(cfg.subtlex_path, cfg.subtlex_sheet_name)

    value_columns: dict[str, str] = {}
    if cfg.subtlex_wf_column and cfg.subtlex_wf_column in subtlex_df.columns:
        value_columns["subtlex_wf"] = cfg.subtlex_wf_column
    if cfg.subtlex_lg10wf_column and cfg.subtlex_lg10wf_column in subtlex_df.columns:
        value_columns["subtlex_lg10wf"] = cfg.subtlex_lg10wf_column
    if cfg.subtlex_cd_column and cfg.subtlex_cd_column in subtlex_df.columns:
        value_columns["subtlex_cd"] = cfg.subtlex_cd_column
    if cfg.subtlex_lg10cd_column and cfg.subtlex_lg10cd_column in subtlex_df.columns:
        value_columns["subtlex_lg10cd"] = cfg.subtlex_lg10cd_column
    if cfg.subtlex_zipf_column and cfg.subtlex_zipf_column in subtlex_df.columns:
        value_columns["subtlex_zipf"] = cfg.subtlex_zipf_column

    pair_maps, word_maps, prepared_df = _build_resource_lookup_maps(
        subtlex_df,
        word_column=cfg.subtlex_word_column,
        pos_column=cfg.subtlex_pos_column,
        value_columns=value_columns,
        cfg=cfg,
        dedupe_score_column="subtlex_wf",
        dedupe_score_ascending=False,
    )

    for feature_name in [
        "subtlex_wf",
        "subtlex_lg10wf",
        "subtlex_cd",
        "subtlex_lg10cd",
        "subtlex_zipf",
    ]:
        if feature_name in pair_maps:
            result[feature_name] = _lookup_with_pos_fallback(
                lookup_word, lookup_pos, pair_maps[feature_name], word_maps[feature_name]
            ).astype(float)

    pair_keys = set()
    word_keys = set()
    if not prepared_df.empty:
        pair_keys = {
            _make_pair_key(word, pos)
            for word, pos in zip(prepared_df["_lookup_word"], prepared_df["_lookup_pos"])
        }
        word_keys = set(prepared_df["_lookup_word"].tolist())

    result["subtlex_found"] = _lookup_found_with_pos_fallback(
        lookup_word, lookup_pos, pair_keys, word_keys
    )
    result["subtlex_pos_match"] = _lookup_pos_exact_match(
        lookup_word, lookup_pos, pair_keys
    )

    return result


def build_features(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    feature_group_cfg: dict[str, Any] | None = None,
    split_name: str = "unknown",
) -> pd.DataFrame:
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg)

    result = df.copy()
    result = _add_pos_features(result, parsed)
    result = _build_kelly_features(result, parsed)
    result = _build_wordfreq_features(result, parsed)
    result = _build_subtlex_features(result, parsed)

    result.attrs["frequency_feature_columns"] = [
        "source_pos_norm",
        "target_pos_norm",
        "source_pos_code",
        "target_pos_code",
        "source_pos_known",
        "target_pos_known",
        "source_target_pos_match",

        "kelly_rank",
        "kelly_rank_percentile",
        "kelly_points",
        "kelly_cefr",
        "kelly_found",
        "kelly_pos_match",

        "wf_value",
        "wf_zipf",
        "wf_cost",
        "wf_percentile",
        "wf_found",

        "subtlex_wf",
        "subtlex_lg10wf",
        "subtlex_cd",
        "subtlex_lg10cd",
        "subtlex_zipf",
        "subtlex_found",
        "subtlex_pos_match",
    ]
    result.attrs["frequency_split_name"] = split_name
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