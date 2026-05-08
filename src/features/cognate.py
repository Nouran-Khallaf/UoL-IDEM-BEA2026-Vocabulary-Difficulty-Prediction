from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import unicodedata

import pandas as pd

from src.core.exceptions import FeatureValidationError


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(slots=True)
class CognateFeatureConfig:
    columns_expected: list[str]
    fillna: dict[str, Any]
    cast: dict[str, str]
    source_word_column: str
    target_word_column: str
    output_column: str
    weighted_levenshtein_column: str
    compute_mode: str
    forbid_negative_values: list[str]
    clip_to_unit_interval: bool
    min_token_length: int
    cognate_lexicon_path: str | None = None

    use_cognet_columns_if_available: bool = True
    cognet_exact_pair_column: str = "cognet_exact_pair"
    cognet_shared_concept_column: str = "cognet_shared_concept"
    cognet_source_found_column: str = "cognet_source_found"
    cognet_target_found_column: str = "cognet_target_found"
    cognet_intersection_count_column: str = "cognet_intersection_count"

    source_has_alternative_column: str = "L1_source_word_has_alternative"
    source_alternative_count_column: str = "L1_source_word_alternative_count"
    source_has_excluded_word_column: str = "L1_source_word_has_excluded_word"

    output_source_has_alternative_column: str = "cognate_source_has_alternative"
    output_source_alternative_count_column: str = "cognate_source_alternative_count"
    output_source_has_excluded_word_column: str = "cognate_source_has_excluded_word"

    weight_char_similarity: float = 0.35
    weight_weighted_levenshtein: float = 0.35
    weight_cognet_signal: float = 0.30


_DEFAULT_EXPECTED = [
    "cognate_sim",
    "weighted_levenshtein_sim",
    "cognet_exact_pair",
    "cognet_shared_concept",
    "cognet_intersection_count",
    "cognate_source_has_alternative",
    "cognate_source_alternative_count",
    "cognate_source_has_excluded_word",
]
_DEFAULT_FILLNA = {
    "cognate_sim": 0.0,
    "weighted_levenshtein_sim": 0.0,
    "cognet_exact_pair": 0,
    "cognet_shared_concept": 0,
    "cognet_intersection_count": 0,
    "cognate_source_has_alternative": 0,
    "cognate_source_alternative_count": 0,
    "cognate_source_has_excluded_word": 0,
}
_DEFAULT_CAST = {
    "cognate_sim": "float",
    "weighted_levenshtein_sim": "float",
    "cognet_exact_pair": "int",
    "cognet_shared_concept": "int",
    "cognet_intersection_count": "int",
    "cognate_source_has_alternative": "int",
    "cognate_source_alternative_count": "int",
    "cognate_source_has_excluded_word": "int",
}
_ALLOWED_COMPUTE_MODES = {"always_compute", "compute_if_missing", "use_existing_only"}


def _ensure_dataframe(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise FeatureValidationError("Cognate feature builder expects a pandas DataFrame.")
    return df


def _get_nested_dict(d: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    value = d.get(key, {})
    return value if isinstance(value, dict) else {}


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureValidationError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", text)
    return text.casefold().strip()


def _strip_nonword(text: str) -> str:
    tokens = _WORD_RE.findall(text)
    return "".join(tokens)


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _jaccard_bigram_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_set = _bigrams(a)
    b_set = _bigrams(b)
    if not a_set or not b_set:
        return 0.0
    union = a_set | b_set
    return float(len(a_set & b_set) / len(union)) if union else 0.0


def _longest_common_subsequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    return float(lcs / max(m, n)) if max(m, n) > 0 else 0.0


def _prefix_suffix_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    max_prefix = 0
    max_len = min(len(a), len(b))
    for i in range(1, max_len + 1):
        if a[:i] == b[:i]:
            max_prefix = i

    max_suffix = 0
    for i in range(1, max_len + 1):
        if a[-i:] == b[-i:]:
            max_suffix = i

    denom = max(len(a), len(b))
    return float((max_prefix + max_suffix) / (2.0 * denom)) if denom > 0 else 0.0


def _char_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    jac = _jaccard_bigram_similarity(a, b)
    lcs = _longest_common_subsequence_ratio(a, b)
    overlap = _prefix_suffix_overlap(a, b)
    score = 0.45 * jac + 0.40 * lcs + 0.15 * overlap
    return float(max(0.0, min(1.0, score)))


def _substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0

    vowels = set("aeiou")
    if a in vowels and b in vowels:
        return 0.5

    related_pairs = {
        ("c", "k"), ("k", "c"),
        ("f", "v"), ("v", "f"),
        ("i", "y"), ("y", "i"),
        ("s", "z"), ("z", "s"),
        ("u", "w"), ("w", "u"),
        ("g", "j"), ("j", "g"),
    }
    if (a, b) in related_pairs:
        return 0.5

    return 1.0


def _weighted_levenshtein_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1.0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1.0

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            bj = b[j - 1]
            delete_cost = dp[i - 1][j] + 1.0
            insert_cost = dp[i][j - 1] + 1.0
            subst_cost = dp[i - 1][j - 1] + _substitution_cost(ai, bj)
            dp[i][j] = min(delete_cost, insert_cost, subst_cost)

    return float(dp[m][n])


def _weighted_levenshtein_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    dist = _weighted_levenshtein_distance(a, b)
    denom = float(max(len(a), len(b)))
    if denom <= 0:
        return 0.0
    sim = 1.0 - (dist / denom)
    return float(max(0.0, min(1.0, sim)))


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception as e:
        raise FeatureValidationError(
            f"Failed to coerce cognate column '{column_name}' to numeric: {e}"
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
        f"Unsupported cognate cast type '{target_type}' for column '{column_name}'."
    )


def _parse_feature_group_cfg(
    feature_group_cfg: dict[str, Any] | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> CognateFeatureConfig:
    feature_group_cfg = feature_group_cfg or {}
    cfg = cfg or {}

    if not isinstance(feature_group_cfg, dict):
        raise FeatureValidationError("feature_group_cfg for cognate features must be a dictionary.")
    if not isinstance(cfg, dict):
        raise FeatureValidationError("cfg for cognate features must be a dictionary when provided.")

    preprocessing = feature_group_cfg.get("preprocessing", {})
    validation = feature_group_cfg.get("validation", {})
    columns_cfg = cfg.get("columns") if isinstance(cfg.get("columns"), dict) else {}

    if preprocessing is None:
        preprocessing = {}
    if validation is None:
        validation = {}
    if not isinstance(preprocessing, dict):
        raise FeatureValidationError("cognate.preprocessing must be a dictionary.")
    if not isinstance(validation, dict):
        raise FeatureValidationError("cognate.validation must be a dictionary.")

    fillna = dict(_DEFAULT_FILLNA)
    fillna.update(_get_nested_dict(preprocessing, "fillna"))

    cast = dict(_DEFAULT_CAST)
    cast.update(_get_nested_dict(preprocessing, "cast"))

    output_column = feature_group_cfg.get("output_column", "cognate_sim")
    output_column = _require_nonempty_string(output_column, "cognate.output_column")

    weighted_levenshtein_column = feature_group_cfg.get("weighted_levenshtein_column", "weighted_levenshtein_sim")
    weighted_levenshtein_column = _require_nonempty_string(
        weighted_levenshtein_column, "cognate.weighted_levenshtein_column"
    )

    output_source_has_alternative_column = feature_group_cfg.get(
        "output_source_has_alternative_column",
        "cognate_source_has_alternative",
    )
    output_source_has_alternative_column = _require_nonempty_string(
        output_source_has_alternative_column,
        "cognate.output_source_has_alternative_column",
    )

    output_source_alternative_count_column = feature_group_cfg.get(
        "output_source_alternative_count_column",
        "cognate_source_alternative_count",
    )
    output_source_alternative_count_column = _require_nonempty_string(
        output_source_alternative_count_column,
        "cognate.output_source_alternative_count_column",
    )

    output_source_has_excluded_word_column = feature_group_cfg.get(
        "output_source_has_excluded_word_column",
        "cognate_source_has_excluded_word",
    )
    output_source_has_excluded_word_column = _require_nonempty_string(
        output_source_has_excluded_word_column,
        "cognate.output_source_has_excluded_word_column",
    )

    default_expected = [
        output_column,
        weighted_levenshtein_column,
        "cognet_exact_pair",
        "cognet_shared_concept",
        "cognet_intersection_count",
        output_source_has_alternative_column,
        output_source_alternative_count_column,
        output_source_has_excluded_word_column,
    ]
    columns_expected = feature_group_cfg.get("columns_expected", default_expected)
    if not isinstance(columns_expected, list):
        raise FeatureValidationError("cognate.columns_expected must be a list.")
    columns_expected = [str(c).strip() for c in columns_expected if str(c).strip()]

    source_word_column = feature_group_cfg.get("source_word_column") or columns_cfg.get("source_word") or "source_word"
    target_word_column = feature_group_cfg.get("target_word_column") or columns_cfg.get("en_word") or "en_target_word"
    source_word_column = _require_nonempty_string(source_word_column, "cognate.source_word_column")
    target_word_column = _require_nonempty_string(target_word_column, "cognate.target_word_column")

    compute_mode = str(feature_group_cfg.get("compute_mode", "always_compute")).strip().lower()
    if compute_mode not in _ALLOWED_COMPUTE_MODES:
        raise FeatureValidationError(
            f"Unsupported cognate.compute_mode '{compute_mode}'. Allowed: {sorted(_ALLOWED_COMPUTE_MODES)}"
        )

    clip_to_unit_interval = bool(validation.get("clip_to_unit_interval", True))

    forbid_negative_values = validation.get(
        "forbid_negative_values",
        [
            output_column,
            weighted_levenshtein_column,
            "cognet_exact_pair",
            "cognet_shared_concept",
            "cognet_intersection_count",
            output_source_has_alternative_column,
            output_source_alternative_count_column,
            output_source_has_excluded_word_column,
        ],
    )
    if forbid_negative_values is None:
        forbid_negative_values = []
    if not isinstance(forbid_negative_values, list):
        raise FeatureValidationError("cognate.validation.forbid_negative_values must be a list.")
    forbid_negative_values = [str(c).strip() for c in forbid_negative_values if str(c).strip()]

    min_token_length = validation.get("min_token_length", 1)
    if not isinstance(min_token_length, int) or min_token_length < 1:
        raise FeatureValidationError("cognate.validation.min_token_length must be an integer >= 1.")

    cognate_lexicon_path = feature_group_cfg.get("cognate_lexicon_path")
    if cognate_lexicon_path is not None and (not isinstance(cognate_lexicon_path, str) or not cognate_lexicon_path.strip()):
        raise FeatureValidationError("cognate.cognate_lexicon_path must be a non-empty string when provided.")

    use_cognet_columns_if_available = bool(feature_group_cfg.get("use_cognet_columns_if_available", True))
    cognet_exact_pair_column = str(feature_group_cfg.get("cognet_exact_pair_column", "cognet_exact_pair")).strip()
    cognet_shared_concept_column = str(feature_group_cfg.get("cognet_shared_concept_column", "cognet_shared_concept")).strip()
    cognet_source_found_column = str(feature_group_cfg.get("cognet_source_found_column", "cognet_source_found")).strip()
    cognet_target_found_column = str(feature_group_cfg.get("cognet_target_found_column", "cognet_target_found")).strip()
    cognet_intersection_count_column = str(feature_group_cfg.get("cognet_intersection_count_column", "cognet_intersection_count")).strip()

    source_has_alternative_column = str(
        feature_group_cfg.get("source_has_alternative_column", "L1_source_word_has_alternative")
    ).strip()
    source_alternative_count_column = str(
        feature_group_cfg.get("source_alternative_count_column", "L1_source_word_alternative_count")
    ).strip()
    source_has_excluded_word_column = str(
        feature_group_cfg.get("source_has_excluded_word_column", "L1_source_word_has_excluded_word")
    ).strip()

    weight_char_similarity = float(feature_group_cfg.get("weight_char_similarity", 0.35))
    weight_weighted_levenshtein = float(feature_group_cfg.get("weight_weighted_levenshtein", 0.35))
    weight_cognet_signal = float(feature_group_cfg.get("weight_cognet_signal", 0.30))

    weight_sum = weight_char_similarity + weight_weighted_levenshtein + weight_cognet_signal
    if weight_sum <= 0:
        raise FeatureValidationError("Cognate feature weights must sum to a positive value.")

    weight_char_similarity /= weight_sum
    weight_weighted_levenshtein /= weight_sum
    weight_cognet_signal /= weight_sum

    return CognateFeatureConfig(
        columns_expected=columns_expected,
        fillna=fillna,
        cast=cast,
        source_word_column=source_word_column,
        target_word_column=target_word_column,
        output_column=output_column,
        weighted_levenshtein_column=weighted_levenshtein_column,
        compute_mode=compute_mode,
        forbid_negative_values=forbid_negative_values,
        clip_to_unit_interval=clip_to_unit_interval,
        min_token_length=min_token_length,
        cognate_lexicon_path=cognate_lexicon_path,
        use_cognet_columns_if_available=use_cognet_columns_if_available,
        cognet_exact_pair_column=cognet_exact_pair_column,
        cognet_shared_concept_column=cognet_shared_concept_column,
        cognet_source_found_column=cognet_source_found_column,
        cognet_target_found_column=cognet_target_found_column,
        cognet_intersection_count_column=cognet_intersection_count_column,
        source_has_alternative_column=source_has_alternative_column,
        source_alternative_count_column=source_alternative_count_column,
        source_has_excluded_word_column=source_has_excluded_word_column,
        output_source_has_alternative_column=output_source_has_alternative_column,
        output_source_alternative_count_column=output_source_alternative_count_column,
        output_source_has_excluded_word_column=output_source_has_excluded_word_column,
        weight_char_similarity=weight_char_similarity,
        weight_weighted_levenshtein=weight_weighted_levenshtein,
        weight_cognet_signal=weight_cognet_signal,
    )


def _fill_missing_columns(df: pd.DataFrame, cfg: CognateFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, fill_value in cfg.fillna.items():
        if col not in result.columns:
            result[col] = fill_value
        else:
            result[col] = result[col].fillna(fill_value)
    return result


def _cast_columns(df: pd.DataFrame, cfg: CognateFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    for col, target_type in cfg.cast.items():
        if col not in result.columns:
            continue
        result[col] = _cast_column(result[col], target_type, col)
    return result


def _load_cognate_lexicon(path_str: str | None) -> dict[tuple[str, str], float]:
    if path_str is None:
        return {}

    path = Path(path_str)
    if not path.exists():
        raise FeatureValidationError(f"Cognate lexicon file does not exist: {path}")

    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise FeatureValidationError(f"Failed to read cognate lexicon from {path}: {e}") from e

    required_options = [
        ("word 1", "word 2"),
        ("source_word", "target_word"),
        ("l1_word", "en_word"),
    ]

    selected_pair: tuple[str, str] | None = None
    for c1, c2 in required_options:
        if c1 in df.columns and c2 in df.columns:
            selected_pair = (c1, c2)
            break

    if selected_pair is None:
        raise FeatureValidationError(
            f"Cognate lexicon at {path} must contain one of the supported column pairs: {required_options}"
        )

    score_col = None
    for candidate in ("cognate_sim", "score", "similarity"):
        if candidate in df.columns:
            score_col = candidate
            break

    lexicon: dict[tuple[str, str], float] = {}
    c1, c2 = selected_pair
    for _, row in df.iterrows():
        w1 = _strip_nonword(_normalize_text(row[c1]))
        w2 = _strip_nonword(_normalize_text(row[c2]))
        if not w1 or not w2:
            continue
        if score_col is None:
            score = _char_similarity(w1, w2)
        else:
            try:
                score = float(row[score_col])
            except Exception:
                score = _char_similarity(w1, w2)
        lexicon[(w1, w2)] = score
    return lexicon


def _safe_numeric_from_row(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return default
    try:
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _compute_cognet_signal(row: pd.Series, cfg: CognateFeatureConfig) -> float:
    if not cfg.use_cognet_columns_if_available:
        return 0.0

    exact = _safe_numeric_from_row(row, cfg.cognet_exact_pair_column, 0.0)
    shared = _safe_numeric_from_row(row, cfg.cognet_shared_concept_column, 0.0)
    src_found = _safe_numeric_from_row(row, cfg.cognet_source_found_column, 0.0)
    tgt_found = _safe_numeric_from_row(row, cfg.cognet_target_found_column, 0.0)
    intersection_count = _safe_numeric_from_row(row, cfg.cognet_intersection_count_column, 0.0)

    signal = 0.0
    if exact > 0:
        signal += 1.0
    elif shared > 0:
        signal += 0.75
    else:
        if src_found > 0 and tgt_found > 0:
            signal += 0.25
        elif src_found > 0 or tgt_found > 0:
            signal += 0.10

    if intersection_count > 1:
        signal += min(0.10, 0.02 * intersection_count)

    return float(max(0.0, min(1.0, signal)))


def _compute_single_row_features(
    row: pd.Series,
    *,
    cfg: CognateFeatureConfig,
    lexicon: dict[tuple[str, str], float] | None = None,
) -> tuple[float, float]:
    source_word = row[cfg.source_word_column]
    target_word = row[cfg.target_word_column]

    source = _strip_nonword(_normalize_text(source_word))
    target = _strip_nonword(_normalize_text(target_word))

    if len(source) < cfg.min_token_length or len(target) < cfg.min_token_length:
        return 0.0, 0.0

    lexicon_score = None
    if lexicon:
        if (source, target) in lexicon:
            lexicon_score = float(lexicon[(source, target)])
        elif (target, source) in lexicon:
            lexicon_score = float(lexicon[(target, source)])

    char_score = _char_similarity(source, target)
    weighted_lev_score = _weighted_levenshtein_similarity(source, target)
    cognet_signal = _compute_cognet_signal(row, cfg)

    cognate_score = (
        cfg.weight_char_similarity * char_score
        + cfg.weight_weighted_levenshtein * weighted_lev_score
        + cfg.weight_cognet_signal * cognet_signal
    )

    if lexicon_score is not None:
        cognate_score = 0.7 * cognate_score + 0.3 * lexicon_score

    return float(max(0.0, min(1.0, weighted_lev_score))), float(max(0.0, min(1.0, cognate_score)))


def _derive_cognate_feature(df: pd.DataFrame, cfg: CognateFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    output_col = cfg.output_column
    weighted_lev_col = cfg.weighted_levenshtein_column

    if cfg.compute_mode == "use_existing_only":
        return result

    if cfg.source_word_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute cognate feature: missing source word column '{cfg.source_word_column}'."
        )
    if cfg.target_word_column not in result.columns:
        raise FeatureValidationError(
            f"Cannot compute cognate feature: missing target word column '{cfg.target_word_column}'."
        )

    if cfg.compute_mode == "compute_if_missing":
        needed = [output_col, weighted_lev_col]
        if all(col in result.columns and not result[col].isna().all() for col in needed):
            return result

    lexicon = _load_cognate_lexicon(cfg.cognate_lexicon_path)

    weighted_lev_vals: list[float] = []
    cognate_vals: list[float] = []

    for _, row in result.iterrows():
        weighted_lev, cognate_score = _compute_single_row_features(
            row,
            cfg=cfg,
            lexicon=lexicon,
        )
        weighted_lev_vals.append(weighted_lev)
        cognate_vals.append(cognate_score)

    result[weighted_lev_col] = weighted_lev_vals
    result[output_col] = cognate_vals
    return result


def _derive_source_metadata_features(df: pd.DataFrame, cfg: CognateFeatureConfig) -> pd.DataFrame:
    result = df.copy()

    if cfg.source_has_alternative_column in result.columns:
        result[cfg.output_source_has_alternative_column] = pd.to_numeric(
            result[cfg.source_has_alternative_column], errors="coerce"
        )
    else:
        result[cfg.output_source_has_alternative_column] = 0

    if cfg.source_alternative_count_column in result.columns:
        result[cfg.output_source_alternative_count_column] = pd.to_numeric(
            result[cfg.source_alternative_count_column], errors="coerce"
        )
    else:
        result[cfg.output_source_alternative_count_column] = 0

    if cfg.source_has_excluded_word_column in result.columns:
        result[cfg.output_source_has_excluded_word_column] = pd.to_numeric(
            result[cfg.source_has_excluded_word_column], errors="coerce"
        )
    else:
        result[cfg.output_source_has_excluded_word_column] = 0

    return result


def _clip_unit_interval(df: pd.DataFrame, cfg: CognateFeatureConfig) -> pd.DataFrame:
    result = df.copy()
    if not cfg.clip_to_unit_interval:
        return result

    for col in [cfg.output_column, cfg.weighted_levenshtein_column]:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            result[col] = numeric.clip(lower=0.0, upper=1.0)

    return result


def _ensure_expected_columns(df: pd.DataFrame, cfg: CognateFeatureConfig, split_name: str) -> None:
    missing = [col for col in cfg.columns_expected if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"Cognate feature group is missing expected columns in split '{split_name}': {missing}"
        )


def _validate_negative_constraints(df: pd.DataFrame, cfg: CognateFeatureConfig, split_name: str) -> None:
    for col in cfg.forbid_negative_values:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0
        if negative_mask.fillna(False).any():
            raise FeatureValidationError(
                f"Cognate column '{col}' contains negative values in split '{split_name}'."
            )


def _collect_feature_columns(df: pd.DataFrame, cfg: CognateFeatureConfig) -> list[str]:
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
    Produced columns:
      - cognate_sim
      - weighted_levenshtein_sim
      - cognet_exact_pair
      - cognet_shared_concept
      - cognet_intersection_count
      - cognate_source_has_alternative
      - cognate_source_alternative_count
      - cognate_source_has_excluded_word

    Notes:
    - CogNet columns are expected to already exist if you enriched the data beforehand.
    - This builder computes weighted_levenshtein_sim and cognate_sim.
    - It also derives numeric cognate-side source metadata features from existing
      L1_source_word_* metadata columns when available.
    """
    df = _ensure_dataframe(df)
    parsed = _parse_feature_group_cfg(feature_group_cfg, cfg=cfg)

    result = df.copy()
    result = _derive_cognate_feature(result, parsed)
    result = _derive_source_metadata_features(result, parsed)
    result = _fill_missing_columns(result, parsed)
    result = _cast_columns(result, parsed)
    result = _clip_unit_interval(result, parsed)

    _ensure_expected_columns(result, parsed, split_name)
    _validate_negative_constraints(result, parsed, split_name)

    cognate_feature_columns = _collect_feature_columns(result, parsed)
    result.attrs["cognate_feature_columns"] = cognate_feature_columns
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