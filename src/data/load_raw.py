from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import ConfigError, DataValidationError


ALL_SPLITS = ("train", "dev", "test")
DEFAULT_REQUIRED_SPLITS = ("train", "dev")
SUPPORTED_TABULAR_SUFFIXES = {".csv", ".tsv", ".txt"}

_MULTI_SPACE_RE = re.compile(r"\s+")
_SPLIT_VARIANTS_RE = re.compile(r"\s*[,;|]\s*")
_GENERIC_PARENS_RE = re.compile(r"\([^)]*\)")
_PLACEHOLDER_TOKEN_RE = re.compile(
    r"^(etw|etwas|jdn|jdn\.|jmd|jmd\.|jemanden|jemandem|jemand)$",
    flags=re.IGNORECASE,
)

_DE_NOT_RE = re.compile(
    r"\((?:\s*)(?:nicht)\s*:\s*([^)]*?)\s*\)",
    flags=re.IGNORECASE,
)
_ES_NOT_RE = re.compile(
    r"\((?:\s*)(?:no|no\s+es|la\s+respuesta\s+no\s+es)\s*:?\s*([^)]*?)\s*\)",
    flags=re.IGNORECASE,
)
_ES_SUFFIX_NOTE_RE = re.compile(
    r"\((?:\s*)(?:la\s+respuesta\s+)?no\s+termina\s+en\s*:?\s*([^)]*?)\s*\)",
    flags=re.IGNORECASE,
)
_ZH_LEXICAL_NOTE_RE = re.compile(
    r"\((?:\s*)(?:不是|不)\s*([^)]*?)\s*\)",
    flags=re.IGNORECASE,
)
_ZH_SUFFIX_NOTE_RE = re.compile(
    r"\((?:\s*)不以\s*([^)]*?)\s*结尾\s*\)",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class SplitLoadResult:
    df: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class DatasetBundle:
    splits: dict[str, SplitLoadResult]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            split_name: {
                "df": result.df,
                "diagnostics": result.diagnostics,
            }
            for split_name, result in self.splits.items()
        }


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a dictionary, got {type(value).__name__}.")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{name}' must be a list, got {type(value).__name__}.")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean, got {type(value).__name__}.")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _raise_data_error(message: str) -> None:
    raise DataValidationError(message)


def _get_optional_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if value is None:
        return {}
    return _require_dict(value, key)


def _normalize_unicode_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return unicodedata.normalize("NFC", value)


def _strip_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.strip()


def _normalize_spaces(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _normalize_l1_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _strip_leading_parenthetical_gloss(candidate: str, *, l1_value: str | None = None) -> str:
    lang = _normalize_l1_value(l1_value)
    candidate = _normalize_spaces(candidate)

    if lang not in {"zh", "cn", "chinese"}:
        return candidate

    candidate = re.sub(r"^\([^)]*\)\s*", "", candidate)
    candidate = re.sub(r"^（[^）]*）\s*", "", candidate)
    return candidate.strip()


def _split_candidates(text: str, *, l1_value: str | None = None) -> list[str]:
    lang = _normalize_l1_value(l1_value)
    if not text:
        return []

    if lang not in {"zh", "cn", "chinese"}:
        return [part.strip() for part in _SPLIT_VARIANTS_RE.split(text) if part.strip()]

    separators = {",", ";", "|", "，", "；", "/", "、"}
    parts: list[str] = []
    current: list[str] = []

    ascii_paren_depth = 0
    fullwidth_paren_depth = 0

    for ch in text:
        if ch == "(":
            ascii_paren_depth += 1
            current.append(ch)
            continue
        if ch == ")":
            ascii_paren_depth = max(0, ascii_paren_depth - 1)
            current.append(ch)
            continue
        if ch == "（":
            fullwidth_paren_depth += 1
            current.append(ch)
            continue
        if ch == "）":
            fullwidth_paren_depth = max(0, fullwidth_paren_depth - 1)
            current.append(ch)
            continue

        inside_parens = ascii_paren_depth > 0 or fullwidth_paren_depth > 0
        if ch in separators and not inside_parens:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def _extract_language_notes(text: str, *, l1_value: str | None = None) -> tuple[str, list[str], str]:
    if not text:
        return text, [], ""

    lang = _normalize_l1_value(l1_value)
    excluded: list[str] = []
    note_type = ""
    text_wo_notes = text

    patterns: list[tuple[re.Pattern[str], str]] = []
    if lang == "de":
        patterns = [(_DE_NOT_RE, "lexical_exclusion")]
    elif lang == "es":
        patterns = [
            (_ES_SUFFIX_NOTE_RE, "suffix_constraint"),
            (_ES_NOT_RE, "lexical_exclusion"),
        ]
    elif lang in {"zh", "cn", "chinese"}:
        patterns = [
            (_ZH_SUFFIX_NOTE_RE, "suffix_constraint"),
            (_ZH_LEXICAL_NOTE_RE, "lexical_exclusion"),
        ]
    else:
        patterns = [(_DE_NOT_RE, "lexical_exclusion")]

    for pattern, current_note_type in patterns:
        matches = list(pattern.finditer(text_wo_notes))
        if not matches:
            continue

        if not note_type:
            note_type = current_note_type

        for match in matches:
            payload = _normalize_spaces(match.group(1))
            if not payload:
                continue
            excluded.extend(_split_candidates(payload, l1_value=l1_value))

        text_wo_notes = pattern.sub("", text_wo_notes)

    text_wo_notes = _normalize_spaces(text_wo_notes)
    return text_wo_notes, excluded, note_type


def _strip_generic_parentheses(text: str, *, l1_value: str | None = None) -> str:
    if not text:
        return text

    lang = _normalize_l1_value(l1_value)
    if lang in {"zh", "cn", "chinese"}:
        return _normalize_spaces(text)

    return _normalize_spaces(_GENERIC_PARENS_RE.sub("", text))


def _tokenize_for_match(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _choose_best_candidate(candidates: list[str], context: Any, *, l1_value: str | None = None) -> str:
    if not candidates:
        return ""

    if not isinstance(context, str) or not context.strip():
        return candidates[0]

    context_text = str(context).strip()
    lang = _normalize_l1_value(l1_value)

    for candidate in candidates:
        cand = _normalize_spaces(candidate)
        if cand and cand in context_text:
            return cand

    if lang in {"zh", "cn", "chinese"}:
        for candidate in candidates:
            cand = _normalize_spaces(candidate)
            if not cand:
                continue
            cleaned = _strip_leading_parenthetical_gloss(cand, l1_value=l1_value)
            if cleaned and cleaned in context_text:
                return cand

    context_lower = context_text.lower()
    context_tokens = set(_tokenize_for_match(context_text))

    for candidate in candidates:
        cand = candidate.strip()
        if not cand:
            continue
        if cand.lower() in context_lower:
            return cand

    for candidate in candidates:
        cand_tokens = _tokenize_for_match(candidate)
        if cand_tokens and any(tok in context_tokens for tok in cand_tokens):
            return candidate

    return candidates[0]


def _choose_excluded_word_for_context(
    excluded_words: list[str],
    context: Any,
    *,
    l1_value: str | None = None,
) -> str:
    if not excluded_words:
        return ""

    if not isinstance(context, str) or not context.strip():
        return ""

    context_text = str(context).strip()
    lang = _normalize_l1_value(l1_value)

    for word in excluded_words:
        cand = _normalize_spaces(word)
        if cand and cand in context_text:
            return cand

    if lang in {"zh", "cn", "chinese"}:
        for word in excluded_words:
            cand = _normalize_spaces(word)
            if not cand:
                continue
            cleaned = _strip_leading_parenthetical_gloss(cand, l1_value=l1_value)
            if cleaned and cleaned in context_text:
                return cand

    context_lower = context_text.lower()
    context_tokens = set(_tokenize_for_match(context_text))

    for word in excluded_words:
        cand = _normalize_spaces(word)
        if not cand:
            continue
        if cand.lower() in context_lower:
            return cand

    for word in excluded_words:
        cand_tokens = _tokenize_for_match(word)
        if cand_tokens and any(tok in context_tokens for tok in cand_tokens):
            return word

    return ""


def _canonicalize_candidate(candidate: str, *, l1_value: str | None = None) -> str:
    candidate = _normalize_spaces(candidate)
    if not candidate:
        return ""

    lang = _normalize_l1_value(l1_value)

    candidate = re.sub(r"-+$", "", candidate).strip()
    if not candidate:
        return ""

    candidate = _strip_leading_parenthetical_gloss(candidate, l1_value=l1_value)
    if not candidate:
        return ""

    if lang in {"zh", "cn", "chinese"}:
        return candidate

    tokens = [tok for tok in re.split(r"\s+", candidate) if tok]
    filtered_tokens = [tok for tok in tokens if not _PLACEHOLDER_TOKEN_RE.fullmatch(tok)]

    if filtered_tokens:
        return filtered_tokens[0]
    if tokens:
        return tokens[0]
    return candidate


def _deduplicate_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_candidates: list[str] = []

    for candidate in candidates:
        cleaned = _normalize_spaces(candidate)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(cleaned)

    return unique_candidates


def _normalize_source_entry(value: Any, context: Any, *, l1_value: str | None = None) -> dict[str, Any]:
    empty_result = {
        "normalized": value if not isinstance(value, str) else "",
        "excluded_words": "",
        "has_excluded_word": 0,
        "excluded_note_type": "",
        "context_excluded_word": "",
        "context_has_excluded_word": 0,
        "has_alternative": 0,
        "alternative_count": 0,
        "alternatives": "",
    }

    if not isinstance(value, str):
        return empty_result

    raw = _normalize_spaces(value)
    if not raw:
        return empty_result

    text_wo_notes, excluded_words, note_type = _extract_language_notes(raw, l1_value=l1_value)
    cleaned = _strip_generic_parentheses(text_wo_notes, l1_value=l1_value)

    context_excluded_word = _choose_excluded_word_for_context(
        excluded_words,
        context,
        l1_value=l1_value,
    )

    if not cleaned:
        return {
            "normalized": "",
            "excluded_words": "|".join(excluded_words),
            "has_excluded_word": int(bool(excluded_words)),
            "excluded_note_type": note_type,
            "context_excluded_word": context_excluded_word,
            "context_has_excluded_word": int(bool(context_excluded_word)),
            "has_alternative": 0,
            "alternative_count": 0,
            "alternatives": "",
        }

    candidates = _split_candidates(cleaned, l1_value=l1_value)
    if not candidates:
        candidates = [cleaned]

    unique_candidates = _deduplicate_candidates(candidates)

    if not unique_candidates:
        return {
            "normalized": "",
            "excluded_words": "|".join(excluded_words),
            "has_excluded_word": int(bool(excluded_words)),
            "excluded_note_type": note_type,
            "context_excluded_word": context_excluded_word,
            "context_has_excluded_word": int(bool(context_excluded_word)),
            "has_alternative": 0,
            "alternative_count": 0,
            "alternatives": "",
        }

    best_candidate = _choose_best_candidate(unique_candidates, context, l1_value=l1_value)
    normalized = _canonicalize_candidate(best_candidate, l1_value=l1_value)
    if not normalized and best_candidate:
        normalized = _normalize_spaces(best_candidate)

    best_candidate_key = _normalize_spaces(best_candidate).casefold()
    remaining_alternatives = [
        cand
        for cand in unique_candidates
        if _normalize_spaces(cand).casefold() != best_candidate_key
    ]

    alternative_count = len(remaining_alternatives)
    has_alternative = int(alternative_count > 0)

    return {
        "normalized": normalized,
        "excluded_words": "|".join(excluded_words),
        "has_excluded_word": int(bool(excluded_words)),
        "excluded_note_type": note_type,
        "context_excluded_word": context_excluded_word,
        "context_has_excluded_word": int(bool(context_excluded_word)),
        "has_alternative": has_alternative,
        "alternative_count": alternative_count,
        "alternatives": "|".join(remaining_alternatives),
    }


def _resolve_text_columns(cfg: dict[str, Any]) -> list[str]:
    schema_cfg = _require_dict(cfg.get("schema"), "schema")

    text_columns: list[str] = []

    schema_text_columns = schema_cfg.get("text_columns")
    if isinstance(schema_text_columns, dict):
        for value in schema_text_columns.values():
            if isinstance(value, str) and value.strip() and value not in text_columns:
                text_columns.append(value.strip())

    columns_cfg = cfg.get("columns")
    if isinstance(columns_cfg, dict):
        for key in ("context", "en_word", "clue", "source_word", "en_pos"):
            value = columns_cfg.get(key)
            if isinstance(value, str) and value.strip() and value not in text_columns:
                text_columns.append(value.strip())

    for key in ("id_column", "l1_column"):
        value = schema_cfg.get(key)
        if isinstance(value, str) and value.strip() and value not in text_columns:
            text_columns.append(value.strip())

    return text_columns


def _normalize_text_columns(
    df: pd.DataFrame,
    *,
    text_columns: list[str],
    strip_whitespace: bool,
    normalize_unicode: bool,
) -> pd.DataFrame:
    df = df.copy()

    for col in text_columns:
        if col not in df.columns:
            continue

        if normalize_unicode:
            df[col] = df[col].map(_normalize_unicode_text)

        if strip_whitespace:
            df[col] = df[col].map(_strip_text)

    return df


def _resolve_source_normalization_columns(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    columns_cfg = cfg.get("columns")
    if not isinstance(columns_cfg, dict):
        return None, None

    source_col = columns_cfg.get("source_word")
    context_col = columns_cfg.get("context")

    source_col_name = source_col.strip() if isinstance(source_col, str) and source_col.strip() else None
    context_col_name = context_col.strip() if isinstance(context_col, str) and context_col.strip() else None
    return source_col_name, context_col_name


def _normalize_source_word_column(
    df: pd.DataFrame,
    *,
    source_col: str | None,
    context_col: str | None,
    l1_col: str | None,
    keep_raw_backup: bool = True,
) -> pd.DataFrame:
    if not source_col or source_col not in df.columns:
        return df

    df = df.copy()

    raw_backup_col = f"{source_col}_raw"
    excluded_col = f"{source_col}_excluded_word"
    has_excluded_col = f"{source_col}_has_excluded_word"
    excluded_note_type_col = f"{source_col}_excluded_note_type"
    context_excluded_col = "L1_context_excluded_word"
    context_has_excluded_col = "L1_context_has_excluded_word"
    has_alternative_col = f"{source_col}_has_alternative"
    alternative_count_col = f"{source_col}_alternative_count"
    alternatives_col = f"{source_col}_alternatives"

    if keep_raw_backup and raw_backup_col not in df.columns:
        df[raw_backup_col] = df[source_col]

    context_series = (
        df[context_col]
        if context_col and context_col in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    l1_series = (
        df[l1_col]
        if l1_col and l1_col in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    normalized_rows = [
        _normalize_source_entry(source_value, context_value, l1_value=l1_value)
        for source_value, context_value, l1_value in zip(
            df[source_col].tolist(),
            context_series.tolist(),
            l1_series.tolist(),
        )
    ]

    df[source_col] = [row["normalized"] for row in normalized_rows]
    df[excluded_col] = [row["excluded_words"] for row in normalized_rows]
    df[has_excluded_col] = [row["has_excluded_word"] for row in normalized_rows]
    df[excluded_note_type_col] = [row["excluded_note_type"] for row in normalized_rows]
    df[context_excluded_col] = [row["context_excluded_word"] for row in normalized_rows]
    df[context_has_excluded_col] = [row["context_has_excluded_word"] for row in normalized_rows]
    df[has_alternative_col] = [row["has_alternative"] for row in normalized_rows]
    df[alternative_count_col] = [row["alternative_count"] for row in normalized_rows]
    df[alternatives_col] = [row["alternatives"] for row in normalized_rows]

    return df


def _cast_series_with_nullable_support(series: pd.Series, target_type: str, column_name: str) -> pd.Series:
    target_type = target_type.strip().lower()

    if target_type in {"str", "string"}:
        return series.astype("string")

    if target_type == "float":
        return pd.to_numeric(series, errors="coerce").astype(float)

    if target_type == "int":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            bad_count = int(numeric.isna().sum())
            _raise_data_error(
                f"Column '{column_name}' cannot be safely cast to int because it contains "
                f"{bad_count} non-numeric or missing values."
            )
        return numeric.astype(int)

    raise ConfigError(f"Unsupported cast type '{target_type}' for column '{column_name}'.")


def _cast_columns(df: pd.DataFrame, cast_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    for col, target_type in cast_map.items():
        if col not in df.columns:
            continue
        df[col] = _cast_series_with_nullable_support(df[col], target_type, col)

    return df


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    split_name: str,
    required_columns: list[str],
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        _raise_data_error(
            f"Missing required columns in {split_name} split: {missing}"
        )


def _validate_unique_ids(
    df: pd.DataFrame,
    *,
    split_name: str,
    id_column: str,
    require_unique: bool,
) -> None:
    if not require_unique:
        return

    if id_column not in df.columns:
        _raise_data_error(
            f"ID column '{id_column}' not found in {split_name} split."
        )

    duplicated_mask = df[id_column].duplicated(keep=False)
    if duplicated_mask.any():
        dup_ids = df.loc[duplicated_mask, id_column].astype(str).tolist()[:20]
        _raise_data_error(
            f"Duplicate IDs found in {split_name} split for '{id_column}'. Preview: {dup_ids}"
        )


def _validate_allowed_l1_values(
    df: pd.DataFrame,
    *,
    split_name: str,
    l1_column: str,
    allowed_l1_values: list[str],
    enforce_single_l1: bool,
) -> None:
    if l1_column not in df.columns:
        _raise_data_error(
            f"L1 column '{l1_column}' not found in {split_name} split."
        )

    observed_values = sorted(set(df[l1_column].dropna().astype(str).tolist()))
    invalid = [val for val in observed_values if val not in allowed_l1_values]

    if invalid:
        _raise_data_error(
            f"Invalid L1 values found in {split_name} split. "
            f"Allowed={allowed_l1_values}, observed={observed_values}, invalid={invalid}"
        )

    if enforce_single_l1 and len(observed_values) > 1:
        _raise_data_error(
            f"More than one L1 value found in {split_name} split: {observed_values}"
        )


def _validate_nonempty_text_columns(
    df: pd.DataFrame,
    *,
    split_name: str,
    columns: list[str],
) -> None:
    for col in columns:
        if col not in df.columns:
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        blank_mask = non_null.astype(str).str.strip().eq("")
        if blank_mask.any():
            _raise_data_error(
                f"Blank strings found in required text column '{col}' for {split_name} split."
            )


def _validate_target_presence(
    df: pd.DataFrame,
    *,
    split_name: str,
    target_column: str,
    required: bool,
) -> None:
    if not required:
        return

    if target_column not in df.columns:
        _raise_data_error(
            f"Target column '{target_column}' missing from {split_name} split."
        )

    nan_mask = df[target_column].isna()
    if nan_mask.any():
        _raise_data_error(
            f"NaN target values found in {split_name} split for '{target_column}': {int(nan_mask.sum())}"
        )


def _build_split_diagnostics(
    df: pd.DataFrame,
    *,
    split_name: str,
    id_column: str,
    target_column: str | None,
    l1_column: str | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "split": split_name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
    }

    if id_column in df.columns:
        diagnostics["n_unique_ids"] = int(df[id_column].nunique())

    if l1_column and l1_column in df.columns:
        diagnostics["l1_values"] = sorted(set(df[l1_column].dropna().astype(str).tolist()))

    if target_column and target_column in df.columns:
        diagnostics["target_nan_count"] = int(df[target_column].isna().sum())

    diagnostics["missing_rate_by_column"] = {
        col: float(df[col].isna().mean()) for col in df.columns
    }

    return diagnostics


def _get_split_requirement_map(cfg: dict[str, Any]) -> dict[str, bool]:
    availability = _get_optional_dict(cfg, "availability")

    if availability:
        return {
            "train": _require_bool(availability.get("train_required", True), "availability.train_required"),
            "dev": _require_bool(availability.get("dev_required", True), "availability.dev_required"),
            "test": _require_bool(availability.get("test_required", False), "availability.test_required"),
        }

    return {
        "train": True,
        "dev": True,
        "test": False,
    }


def _should_load_split(cfg: dict[str, Any], split_name: str) -> bool:
    files = _require_dict(cfg["files"], "files")
    required_map = _get_split_requirement_map(cfg)

    file_path_value = files.get(split_name, None)
    is_required = required_map[split_name]

    if file_path_value is None:
        return False

    if isinstance(file_path_value, str) and not file_path_value.strip():
        if is_required:
            raise ConfigError(
                f"Split '{split_name}' is required but files.{split_name} is empty."
            )
        return False

    return True


def _read_table(file_path: Path, *, split_name: str) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ConfigError(
            f"Unsupported file type for split '{split_name}': {file_path}. "
            f"Supported suffixes: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
        )

    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        return pd.read_csv(file_path, sep="\t")
    except Exception as e:
        _raise_data_error(
            f"Failed to read table for split '{split_name}' from {file_path}: {e}"
        )
        raise


def _resolve_required_columns(cfg: dict[str, Any], split_name: str) -> list[str]:
    required_columns_cfg = _get_optional_dict(cfg, "required_columns")
    value = required_columns_cfg.get(split_name, [])
    if value is None:
        return []
    value = _require_list(value, f"required_columns.{split_name}")
    return [_require_nonempty_string(v, f"required_columns.{split_name}[]") for v in value]


def _resolve_cast_map(cfg: dict[str, Any]) -> dict[str, str]:
    typing_cfg = _get_optional_dict(cfg, "typing")
    cast_map = typing_cfg.get("cast_columns", {})
    if cast_map is None:
        return {}
    cast_map = _require_dict(cast_map, "typing.cast_columns")
    normalized: dict[str, str] = {}
    for col, dtype in cast_map.items():
        if not isinstance(col, str) or not col.strip():
            raise ConfigError("All keys in typing.cast_columns must be non-empty strings.")
        normalized[col.strip()] = _require_nonempty_string(dtype, f"typing.cast_columns.{col}")
    return normalized


def _apply_optional_filters(df: pd.DataFrame, *, cfg: dict[str, Any], split_name: str) -> pd.DataFrame:
    filters_cfg = _get_optional_dict(cfg, "filters")
    schema_cfg = _require_dict(cfg["schema"], "schema")
    l1_column = _require_nonempty_string(schema_cfg.get("l1_column"), "schema.l1_column")

    result = df.copy()

    keep_only_l1 = filters_cfg.get("keep_only_l1")
    if keep_only_l1 is not None:
        keep_only_l1 = _require_nonempty_string(keep_only_l1, "filters.keep_only_l1")
        if l1_column not in result.columns:
            _raise_data_error(
                f"Cannot filter by L1 because column '{l1_column}' is missing in split '{split_name}'."
            )
        result = result[result[l1_column].astype(str) == keep_only_l1].reset_index(drop=True)
        if result.empty:
            _raise_data_error(
                f"All rows were removed after keep_only_l1='{keep_only_l1}' for split '{split_name}'."
            )

    drop_duplicates_on = filters_cfg.get("drop_duplicates_on")
    if drop_duplicates_on is not None:
        drop_duplicates_on = _require_list(drop_duplicates_on, "filters.drop_duplicates_on")
        subset = [_require_nonempty_string(col, "filters.drop_duplicates_on[]") for col in drop_duplicates_on]
        existing_subset = [col for col in subset if col in result.columns]
        if existing_subset:
            result = result.drop_duplicates(subset=existing_subset, keep="first").reset_index(drop=True)

    return result


def _load_one_split(
    cfg: dict[str, Any],
    *,
    split_name: str,
) -> SplitLoadResult:
    if split_name not in ALL_SPLITS:
        raise ConfigError(f"Unsupported split '{split_name}'. Expected one of {ALL_SPLITS}.")

    files = _require_dict(cfg["files"], "files")
    validation_cfg = _get_optional_dict(cfg, "validation")
    filters_cfg = _get_optional_dict(cfg, "filters")
    schema_cfg = _require_dict(cfg["schema"], "schema")

    file_path_str = files.get(split_name)
    if file_path_str is None:
        raise ConfigError(f"No file configured for split '{split_name}'.")

    file_path = Path(_require_nonempty_string(file_path_str, f"files.{split_name}"))
    if not file_path.exists():
        _raise_data_error(
            f"Configured file does not exist for split '{split_name}': {file_path}"
        )

    df = _read_table(file_path, split_name=split_name)
    if df.empty:
        _raise_data_error(
            f"Loaded dataframe is empty for split '{split_name}'."
        )

    required_columns = _resolve_required_columns(cfg, split_name)
    if required_columns:
        _validate_required_columns(df, split_name=split_name, required_columns=required_columns)

    text_columns = _resolve_text_columns(cfg)
    strip_whitespace = _require_bool(
        filters_cfg.get("strip_whitespace", True),
        "filters.strip_whitespace",
    )
    normalize_unicode = _require_bool(
        filters_cfg.get("normalize_unicode", True),
        "filters.normalize_unicode",
    )

    df = _normalize_text_columns(
        df,
        text_columns=text_columns,
        strip_whitespace=strip_whitespace,
        normalize_unicode=normalize_unicode,
    )

    source_col, context_col = _resolve_source_normalization_columns(cfg)
    l1_column = _require_nonempty_string(schema_cfg.get("l1_column"), "schema.l1_column")
    df = _normalize_source_word_column(
        df,
        source_col=source_col,
        context_col=context_col,
        l1_col=l1_column,
        keep_raw_backup=True,
    )

    cast_map = _resolve_cast_map(cfg)
    if cast_map:
        df = _cast_columns(df, cast_map)

    df = _apply_optional_filters(df, cfg=cfg, split_name=split_name)

    id_column = _require_nonempty_string(schema_cfg.get("id_column"), "schema.id_column")
    l1_column = _require_nonempty_string(schema_cfg.get("l1_column"), "schema.l1_column")
    target_column = _require_nonempty_string(schema_cfg.get("target_column"), "schema.target_column")

    allowed_l1_values = validation_cfg.get("allowed_l1_values", [])
    if not isinstance(allowed_l1_values, list) or not allowed_l1_values:
        raise ConfigError("'validation.allowed_l1_values' must be a non-empty list.")

    enforce_single_l1 = _require_bool(
        validation_cfg.get("enforce_single_l1", True),
        "validation.enforce_single_l1",
    )
    require_nonempty_text = _require_bool(
        validation_cfg.get("require_nonempty_text_columns", True),
        "validation.require_nonempty_text_columns",
    )
    require_unique_ids = _require_bool(
        validation_cfg.get("require_unique_item_id_within_file", True),
        "validation.require_unique_item_id_within_file",
    )

    require_target_map = {
        "train": _require_bool(
            validation_cfg.get("require_target_in_train", True),
            "validation.require_target_in_train",
        ),
        "dev": _require_bool(
            validation_cfg.get("require_target_in_dev", True),
            "validation.require_target_in_dev",
        ),
        "test": _require_bool(
            validation_cfg.get("require_target_in_test", False),
            "validation.require_target_in_test",
        ),
    }

    forbid_blank_columns = validation_cfg.get("forbid_blank_strings_in", [])
    if not isinstance(forbid_blank_columns, list):
        raise ConfigError("'validation.forbid_blank_strings_in' must be a list.")

    _validate_allowed_l1_values(
        df,
        split_name=split_name,
        l1_column=l1_column,
        allowed_l1_values=[str(x) for x in allowed_l1_values],
        enforce_single_l1=enforce_single_l1,
    )

    _validate_unique_ids(
        df,
        split_name=split_name,
        id_column=id_column,
        require_unique=require_unique_ids,
    )

    if require_nonempty_text:
        _validate_nonempty_text_columns(
            df,
            split_name=split_name,
            columns=[col for col in text_columns if col in df.columns],
        )

    if forbid_blank_columns:
        _validate_nonempty_text_columns(
            df,
            split_name=split_name,
            columns=[str(col) for col in forbid_blank_columns],
        )

    _validate_target_presence(
        df,
        split_name=split_name,
        target_column=target_column,
        required=require_target_map[split_name],
    )

    diagnostics = _build_split_diagnostics(
        df,
        split_name=split_name,
        id_column=id_column,
        target_column=target_column,
        l1_column=l1_column,
    )
    diagnostics["file_path"] = str(file_path.resolve())

    return SplitLoadResult(df=df, diagnostics=diagnostics)


def load_raw_dataset(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Load raw datasets according to the resolved config.

    Supports:
    - train/dev only
    - train/dev/test later when test becomes available

    Returns
    -------
    dict
        Example:
        {
          "train": {"df": <pd.DataFrame>, "diagnostics": {...}},
          "dev":   {"df": <pd.DataFrame>, "diagnostics": {...}}
        }
    """
    cfg = _require_dict(cfg, "resolved_config")
    files = _require_dict(cfg.get("files"), "files")
    required_map = _get_split_requirement_map(cfg)

    loaded: dict[str, SplitLoadResult] = {}

    for split_name in ALL_SPLITS:
        should_load = _should_load_split(cfg, split_name)

        if not should_load:
            if required_map[split_name]:
                _raise_data_error(
                    f"Required split '{split_name}' is not configured. Current value: {files.get(split_name)!r}"
                )
            continue

        loaded[split_name] = _load_one_split(cfg, split_name=split_name)

    for split_name in DEFAULT_REQUIRED_SPLITS:
        if required_map[split_name] and split_name not in loaded:
            _raise_data_error(
                f"Required split '{split_name}' was not loaded."
            )

    return DatasetBundle(splits=loaded).to_dict()