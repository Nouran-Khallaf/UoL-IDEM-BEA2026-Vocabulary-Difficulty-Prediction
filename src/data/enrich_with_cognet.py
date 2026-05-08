from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import load_and_resolve_config
from src.data.load_raw import load_raw_dataset


LANG_CODE_MAP: dict[str, list[str]] = {
    "es": ["spa"],
    "de": ["deu"],
    "cn": ["cmn", "zho"],
    "zh": ["cmn", "zho"],
    "spa": ["spa"],
    "deu": ["deu"],
    "cmn": ["cmn", "zho"],
    "zho": ["zho", "cmn"],
    "eng": ["eng"],
    "en": ["eng"],
}

COGNET_OUTPUT_COLUMNS = [
    "cognet_exact_pair",
    "cognet_shared_concept",
    "cognet_source_found",
    "cognet_target_found",
    "cognet_source_concept_count",
    "cognet_target_concept_count",
    "cognet_intersection_count",
    "cognet_match_type",
    "cognet_best_concept_id",
    "cognet_source_has_alternative_english",
    "cognet_source_alternative_english_count",
    "cognet_source_alternative_english_words",
    "cognet_row_has_any_excluded_word",
]

REQUIRED_COGNET_COLUMNS = [
    "concept_id",
    "lang_1",
    "word_1",
    "lang_2",
    "word_2",
    "translit_1",
    "translit_2",
]

_WS_RE = re.compile(r"\s+")
_ASCII_SPLIT_RE = re.compile(r"\s*[,;|/]\s*")
_GENERIC_PARENS_RE = re.compile(r"\([^)]*\)")
_GERMAN_PLACEHOLDER_RE = re.compile(
    r"^(etw|etwas|jdn|jdn\.|jmd|jmd\.|jemanden|jemandem|jemand)$",
    flags=re.IGNORECASE,
)
_GERMAN_FUNCTION_WORD_RE = re.compile(
    r"^(sich|gegen|an|auf|bei|mit|von|zu|in|im|am|um|für|fuer|durch|nach|vor|über|ueber|unter|aus)$",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class RowMatchResult:
    cognet_exact_pair: int
    cognet_shared_concept: int
    cognet_source_found: int
    cognet_target_found: int
    cognet_source_concept_count: int
    cognet_target_concept_count: int
    cognet_intersection_count: int
    cognet_match_type: str
    cognet_best_concept_id: str
    cognet_source_has_alternative_english: int
    cognet_source_alternative_english_count: int
    cognet_source_alternative_english_words: str
    cognet_row_has_any_excluded_word: int


@dataclass(slots=True)
class DetailedMatch:
    item_id: str
    row_index: int
    l1_raw: str
    source_lang_codes: str
    target_lang_code: str
    source_word: str
    target_word: str
    source_candidates: str
    concept_id: str
    cognet_lang_1: str
    cognet_word_1: str
    cognet_lang_2: str
    cognet_word_2: str
    translit_1: str
    translit_2: str
    match_type: str


def normalize_text(value: Any, *, lowercase: bool = True) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


def normalize_lookup(value: Any) -> str:
    return normalize_text(value, lowercase=True)


def normalize_l1_code(value: Any) -> str:
    return normalize_text(value, lowercase=True)


def split_variants(text: str, *, l1_value: str | None = None) -> list[str]:
    lang = normalize_l1_code(l1_value)
    if not text:
        return []

    if lang in {"zh", "cn", "chinese"}:
        separators = {",", ";", "|", "/", "，", "；", "、"}
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

    return [part.strip() for part in _ASCII_SPLIT_RE.split(text) if part.strip()]


def strip_leading_parenthetical_gloss(candidate: str, *, l1_value: str | None = None) -> str:
    lang = normalize_l1_code(l1_value)
    candidate = normalize_text(candidate, lowercase=False)

    if lang not in {"zh", "cn", "chinese"}:
        return candidate

    candidate = re.sub(r"^\([^)]*\)\s*", "", candidate)
    candidate = re.sub(r"^（[^）]*）\s*", "", candidate)
    return candidate.strip()


def canonicalize_source_candidate(candidate: str, *, l1_value: str | None = None) -> str:
    candidate = normalize_text(candidate, lowercase=False)
    if not candidate:
        return ""

    lang = normalize_l1_code(l1_value)

    candidate = re.sub(r"-+$", "", candidate).strip()
    if not candidate:
        return ""

    candidate = strip_leading_parenthetical_gloss(candidate, l1_value=l1_value)
    if not candidate:
        return ""

    if lang in {"zh", "cn", "chinese"}:
        return candidate

    tokens = [tok for tok in re.split(r"\s+", candidate) if tok]
    if not tokens:
        return ""

    if lang == "de":
        content_tokens = [
            tok for tok in tokens
            if not _GERMAN_PLACEHOLDER_RE.fullmatch(tok)
            and not _GERMAN_FUNCTION_WORD_RE.fullmatch(tok)
        ]
        if content_tokens:
            return content_tokens[-1]

    filtered_tokens = [tok for tok in tokens if not _GERMAN_PLACEHOLDER_RE.fullmatch(tok)]
    if filtered_tokens:
        return filtered_tokens[0]

    return tokens[0]


def extract_source_candidates(
    *,
    source_word: Any,
    source_word_raw: Any,
    source_word_alternatives: Any = None,
    l1_value: Any,
) -> list[str]:
    lang = normalize_l1_code(l1_value)
    candidates: list[str] = []

    def _add(raw_value: Any, *, strip_generic_parens: bool) -> None:
        if raw_value is None:
            return

        text = normalize_text(raw_value, lowercase=False)
        if not text:
            return

        if strip_generic_parens and lang not in {"zh", "cn", "chinese"}:
            text = _GENERIC_PARENS_RE.sub("", text)
            text = normalize_text(text, lowercase=False)

        parts = split_variants(text, l1_value=lang)
        if not parts:
            parts = [text]

        for part in parts:
            part = canonicalize_source_candidate(part, l1_value=lang)
            if part:
                candidates.append(part)

    # priority:
    # 1) cleaned main source word
    # 2) stored alternatives from loader
    # 3) raw source as fallback
    _add(source_word, strip_generic_parens=False)
    _add(source_word_alternatives, strip_generic_parens=False)
    _add(source_word_raw, strip_generic_parens=True)

    deduped: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        norm = normalize_lookup(cand)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(cand)

    return deduped


def resolve_dataset_lang_to_cognet_codes(raw_lang: Any) -> list[str]:
    key = normalize_text(raw_lang, lowercase=True)
    if not key:
        return []
    return LANG_CODE_MAP.get(key, [key])


def load_cognet_subset(cognet_path: Path) -> pd.DataFrame:
    suffix = cognet_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(cognet_path, dtype=str, keep_default_na=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(cognet_path)
        df = df.fillna("")
        for col in df.columns:
            df[col] = df[col].astype(str)
    else:
        raise ValueError(
            f"Unsupported CogNet subset format '{cognet_path.suffix}'. Use .csv or .parquet."
        )

    missing = [c for c in REQUIRED_COGNET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CogNet subset is missing required columns: {missing}")

    df = df.copy()

    for col in REQUIRED_COGNET_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    df["lang_1"] = df["lang_1"].str.strip().str.lower()
    df["lang_2"] = df["lang_2"].str.strip().str.lower()

    if "word_1_norm" not in df.columns:
        df["word_1_norm"] = df["word_1"].map(normalize_lookup)
    if "word_2_norm" not in df.columns:
        df["word_2_norm"] = df["word_2"].map(normalize_lookup)
    if "translit_1_norm" not in df.columns:
        df["translit_1_norm"] = df["translit_1"].map(normalize_lookup)
    if "translit_2_norm" not in df.columns:
        df["translit_2_norm"] = df["translit_2"].map(normalize_lookup)

    print(
        json.dumps(
            {
                "cognet_subset_path": str(cognet_path),
                "n_rows_loaded": int(len(df)),
                "format": suffix.lstrip("."),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return df


def build_indexes(cognet_df: pd.DataFrame) -> dict[str, Any]:
    pair_index: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    word_to_concepts: dict[tuple[str, str], set[str]] = defaultdict(set)
    concept_to_rows: dict[str, list[int]] = defaultdict(list)

    for idx, row in cognet_df.iterrows():
        concept_id = row["concept_id"]
        lang_1 = row["lang_1"]
        lang_2 = row["lang_2"]
        word_1_norm = row["word_1_norm"]
        word_2_norm = row["word_2_norm"]
        translit_1_norm = row["translit_1_norm"]
        translit_2_norm = row["translit_2_norm"]

        concept_to_rows[concept_id].append(idx)

        candidates_1 = {w for w in [word_1_norm, translit_1_norm] if w}
        candidates_2 = {w for w in [word_2_norm, translit_2_norm] if w}

        for w1 in candidates_1:
            word_to_concepts[(lang_1, w1)].add(concept_id)
        for w2 in candidates_2:
            word_to_concepts[(lang_2, w2)].add(concept_id)

        for w1 in candidates_1:
            for w2 in candidates_2:
                pair_index[(lang_1, w1, lang_2, w2)].append(idx)
                pair_index[(lang_2, w2, lang_1, w1)].append(idx)

    return {
        "pair_index": pair_index,
        "word_to_concepts": word_to_concepts,
        "concept_to_rows": concept_to_rows,
    }


def find_exact_pair_matches_any_source(
    *,
    pair_index: dict[tuple[str, str, str, str], list[int]],
    source_lang_codes: list[str],
    target_lang_code: str,
    source_word_norms: list[str],
    target_word_norm: str,
) -> list[int]:
    matched_rows: list[int] = []
    if not source_word_norms or not target_word_norm:
        return matched_rows

    for source_word_norm in source_word_norms:
        for src_lang in source_lang_codes:
            matched_rows.extend(
                pair_index.get((src_lang, source_word_norm, target_lang_code, target_word_norm), [])
            )

    return list(dict.fromkeys(matched_rows))


def get_concepts_for_any_source_word(
    *,
    word_to_concepts: dict[tuple[str, str], set[str]],
    lang_codes: list[str],
    word_norms: list[str],
) -> set[str]:
    concepts: set[str] = set()
    for word_norm in word_norms:
        if not word_norm:
            continue
        for code in lang_codes:
            concepts.update(word_to_concepts.get((code, word_norm), set()))
    return concepts


def get_concepts_for_word(
    *,
    word_to_concepts: dict[tuple[str, str], set[str]],
    lang_codes: list[str],
    word_norm: str,
) -> set[str]:
    if not word_norm:
        return set()

    concepts: set[str] = set()
    for code in lang_codes:
        concepts.update(word_to_concepts.get((code, word_norm), set()))
    return concepts


def classify_row_match(
    *,
    exact_pair_rows: list[int],
    source_concepts: set[str],
    target_concepts: set[str],
) -> tuple[str, str]:
    intersection = source_concepts & target_concepts

    if exact_pair_rows:
        best_concept = sorted(intersection)[0] if intersection else ""
        return "exact_pair", best_concept

    if intersection:
        return "shared_concept_only", sorted(intersection)[0]

    if source_concepts and not target_concepts:
        return "source_only_found", ""

    if target_concepts and not source_concepts:
        return "target_only_found", ""

    if source_concepts and target_concepts:
        return "both_found_no_shared_concept", ""

    return "no_match", ""


def get_alternative_english_words_for_source(
    *,
    cognet_df: pd.DataFrame,
    concept_to_rows: dict[str, list[int]],
    source_concepts: set[str],
    target_lang_code: str,
    target_word_norm: str,
    source_word_norms: list[str],
) -> tuple[int, list[str]]:
    alternative_words: set[str] = set()
    source_norm_set = {w for w in source_word_norms if w}

    for concept_id in source_concepts:
        row_ids = concept_to_rows.get(concept_id, [])
        for cog_idx in row_ids:
            row = cognet_df.loc[cog_idx]

            if row["lang_1"] == target_lang_code:
                for candidate in [row["word_1_norm"], row["translit_1_norm"]]:
                    if candidate:
                        alternative_words.add(candidate)

            if row["lang_2"] == target_lang_code:
                for candidate in [row["word_2_norm"], row["translit_2_norm"]]:
                    if candidate:
                        alternative_words.add(candidate)

    if target_word_norm:
        alternative_words.discard(target_word_norm)

    alternative_words = {w for w in alternative_words if w not in source_norm_set}

    alternatives_sorted = sorted(alternative_words)
    return int(bool(alternatives_sorted)), alternatives_sorted


def make_row_result(
    *,
    exact_pair_rows: list[int],
    source_concepts: set[str],
    target_concepts: set[str],
    source_has_alternative_english: int,
    source_alternative_english_words: list[str],
    context_excluded_words: list[str],
    source_word_excluded_words: list[str],
) -> RowMatchResult:
    match_type, best_concept_id = classify_row_match(
        exact_pair_rows=exact_pair_rows,
        source_concepts=source_concepts,
        target_concepts=target_concepts,
    )

    intersection = source_concepts & target_concepts
    row_has_any_excluded = int(bool(context_excluded_words or source_word_excluded_words))

    return RowMatchResult(
        cognet_exact_pair=int(bool(exact_pair_rows)),
        cognet_shared_concept=int(bool(intersection)),
        cognet_source_found=int(bool(source_concepts)),
        cognet_target_found=int(bool(target_concepts)),
        cognet_source_concept_count=int(len(source_concepts)),
        cognet_target_concept_count=int(len(target_concepts)),
        cognet_intersection_count=int(len(intersection)),
        cognet_match_type=match_type,
        cognet_best_concept_id=best_concept_id,
        cognet_source_has_alternative_english=int(source_has_alternative_english),
        cognet_source_alternative_english_count=int(len(source_alternative_english_words)),
        cognet_source_alternative_english_words="|".join(source_alternative_english_words),
        cognet_row_has_any_excluded_word=row_has_any_excluded,
    )


def build_detailed_matches(
    *,
    cognet_df: pd.DataFrame,
    row_index: int,
    item_id: str,
    l1_raw: str,
    source_lang_codes: list[str],
    target_lang_code: str,
    source_word: str,
    target_word: str,
    source_candidates: list[str],
    exact_pair_rows: list[int],
    source_concepts: set[str],
    target_concepts: set[str],
) -> list[DetailedMatch]:
    detailed: list[DetailedMatch] = []

    chosen_row_ids: list[int] = []
    if exact_pair_rows:
        chosen_row_ids = exact_pair_rows
        match_type = "exact_pair"
    else:
        shared = source_concepts & target_concepts
        if shared:
            concept_to_row: dict[str, int] = {}
            for cog_idx in cognet_df.index:
                cid = cognet_df.at[cog_idx, "concept_id"]
                if cid in shared and cid not in concept_to_row:
                    concept_to_row[cid] = cog_idx
            chosen_row_ids = list(concept_to_row.values())
            match_type = "shared_concept_only"
        else:
            return detailed

    for cog_idx in chosen_row_ids:
        row = cognet_df.loc[cog_idx]
        detailed.append(
            DetailedMatch(
                item_id=item_id,
                row_index=row_index,
                l1_raw=l1_raw,
                source_lang_codes="|".join(source_lang_codes),
                target_lang_code=target_lang_code,
                source_word=source_word,
                target_word=target_word,
                source_candidates="|".join(source_candidates),
                concept_id=row["concept_id"],
                cognet_lang_1=row["lang_1"],
                cognet_word_1=row["word_1"],
                cognet_lang_2=row["lang_2"],
                cognet_word_2=row["word_2"],
                translit_1=row["translit_1"],
                translit_2=row["translit_2"],
                match_type=match_type,
            )
        )

    return detailed


def enrich_dataframe_with_cognet(
    df: pd.DataFrame,
    *,
    cognet_df: pd.DataFrame,
    indexes: dict[str, Any],
    lang_column: str,
    source_word_column: str,
    target_word_column: str,
    target_lang_code: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = [lang_column, source_word_column, target_word_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    if id_column not in df.columns:
        df = df.copy()
        df[id_column] = [str(i) for i in range(len(df))]

    pair_index = indexes["pair_index"]
    word_to_concepts = indexes["word_to_concepts"]
    concept_to_rows = indexes["concept_to_rows"]

    enriched = df.copy()
    existing_cognet_cols = [c for c in COGNET_OUTPUT_COLUMNS if c in enriched.columns]
    if existing_cognet_cols:
        enriched = enriched.drop(columns=existing_cognet_cols)

    row_results: list[RowMatchResult] = []
    detailed_matches: list[DetailedMatch] = []

    unmatched_source_counter: Counter[str] = Counter()
    unmatched_target_counter: Counter[str] = Counter()
    per_lang_counts: dict[str, Counter[str]] = defaultdict(Counter)

    source_word_raw_column = f"{source_word_column}_raw"
    source_word_excluded_column = f"{source_word_column}_excluded_word"
    source_word_alternatives_column = f"{source_word_column}_alternatives"
    context_excluded_column = "L1_context_excluded_word"
    context_has_excluded_column = "L1_context_has_excluded_word"

    for row_index, row in enriched.iterrows():
        item_id = str(row[id_column])
        l1_raw = str(row[lang_column])
        source_lang_codes = resolve_dataset_lang_to_cognet_codes(l1_raw)

        source_word = str(row[source_word_column]) if pd.notna(row[source_word_column]) else ""
        target_word = str(row[target_word_column]) if pd.notna(row[target_word_column]) else ""

        source_word_raw = (
            str(row[source_word_raw_column])
            if source_word_raw_column in enriched.columns and pd.notna(row[source_word_raw_column])
            else source_word
        )

        source_word_alternatives = (
            str(row[source_word_alternatives_column])
            if source_word_alternatives_column in enriched.columns and pd.notna(row[source_word_alternatives_column])
            else ""
        )

        source_word_excluded_words: list[str] = []
        if source_word_excluded_column in enriched.columns and pd.notna(row[source_word_excluded_column]):
            source_word_excluded_words = [
                normalize_lookup(x)
                for x in str(row[source_word_excluded_column]).split("|")
                if normalize_lookup(x)
            ]

        context_excluded_words: list[str] = []
        if context_excluded_column in enriched.columns and pd.notna(row[context_excluded_column]):
            context_excluded_words = [
                normalize_lookup(x)
                for x in str(row[context_excluded_column]).split("|")
                if normalize_lookup(x)
            ]

        if context_has_excluded_column in enriched.columns and pd.notna(row[context_has_excluded_column]):
            try:
                context_has_excluded_word = int(row[context_has_excluded_column])
            except Exception:
                context_has_excluded_word = int(bool(context_excluded_words))
        else:
            context_has_excluded_word = int(bool(context_excluded_words))

        if context_has_excluded_word == 0:
            context_excluded_words = []

        source_candidates = extract_source_candidates(
            source_word=source_word,
            source_word_raw=source_word_raw,
            source_word_alternatives=source_word_alternatives,
            l1_value=l1_raw,
        )
        source_word_norms = [normalize_lookup(x) for x in source_candidates if normalize_lookup(x)]
        target_word_norm = normalize_lookup(target_word)

        if not source_lang_codes:
            result = RowMatchResult(
                cognet_exact_pair=0,
                cognet_shared_concept=0,
                cognet_source_found=0,
                cognet_target_found=0,
                cognet_source_concept_count=0,
                cognet_target_concept_count=0,
                cognet_intersection_count=0,
                cognet_match_type="unknown_source_language",
                cognet_best_concept_id="",
                cognet_source_has_alternative_english=0,
                cognet_source_alternative_english_count=0,
                cognet_source_alternative_english_words="",
                cognet_row_has_any_excluded_word=int(bool(context_excluded_words or source_word_excluded_words)),
            )
            row_results.append(result)
            per_lang_counts[normalize_lookup(l1_raw)]["unknown_source_language"] += 1
            for source_word_norm in source_word_norms:
                unmatched_source_counter[source_word_norm] += 1
            if target_word_norm:
                unmatched_target_counter[target_word_norm] += 1
            continue

        exact_pair_rows = find_exact_pair_matches_any_source(
            pair_index=pair_index,
            source_lang_codes=source_lang_codes,
            target_lang_code=target_lang_code,
            source_word_norms=source_word_norms,
            target_word_norm=target_word_norm,
        )

        source_concepts = get_concepts_for_any_source_word(
            word_to_concepts=word_to_concepts,
            lang_codes=source_lang_codes,
            word_norms=source_word_norms,
        )
        target_concepts = get_concepts_for_word(
            word_to_concepts=word_to_concepts,
            lang_codes=[target_lang_code],
            word_norm=target_word_norm,
        )

        source_has_alternative_english, alternative_english_words = get_alternative_english_words_for_source(
            cognet_df=cognet_df,
            concept_to_rows=concept_to_rows,
            source_concepts=source_concepts,
            target_lang_code=target_lang_code,
            target_word_norm=target_word_norm,
            source_word_norms=source_word_norms,
        )

        result = make_row_result(
            exact_pair_rows=exact_pair_rows,
            source_concepts=source_concepts,
            target_concepts=target_concepts,
            source_has_alternative_english=source_has_alternative_english,
            source_alternative_english_words=alternative_english_words,
            context_excluded_words=context_excluded_words,
            source_word_excluded_words=source_word_excluded_words,
        )
        row_results.append(result)
        per_lang_counts[normalize_lookup(l1_raw)][result.cognet_match_type] += 1

        if not result.cognet_source_found:
            for source_word_norm in source_word_norms:
                unmatched_source_counter[source_word_norm] += 1

        if not result.cognet_target_found and target_word_norm:
            unmatched_target_counter[target_word_norm] += 1

        detailed_matches.extend(
            build_detailed_matches(
                cognet_df=cognet_df,
                row_index=row_index,
                item_id=item_id,
                l1_raw=l1_raw,
                source_lang_codes=source_lang_codes,
                target_lang_code=target_lang_code,
                source_word=source_word,
                target_word=target_word,
                source_candidates=source_candidates,
                exact_pair_rows=exact_pair_rows,
                source_concepts=source_concepts,
                target_concepts=target_concepts,
            )
        )

    result_df = pd.DataFrame([asdict(r) for r in row_results], index=enriched.index)
    enriched = pd.concat([enriched, result_df], axis=1)

    if enriched.columns.duplicated().any():
        dupes = enriched.columns[enriched.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns found after CogNet enrichment: {dupes}")

    matches_df = pd.DataFrame([asdict(m) for m in detailed_matches])

    stats = build_stats(
        enriched_df=enriched,
        matches_df=matches_df,
        lang_column=lang_column,
        source_word_column=source_word_column,
        target_word_column=target_word_column,
        unmatched_source_counter=unmatched_source_counter,
        unmatched_target_counter=unmatched_target_counter,
        per_lang_counts=per_lang_counts,
    )
    return enriched, matches_df, stats


def build_stats(
    *,
    enriched_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    lang_column: str,
    source_word_column: str,
    target_word_column: str,
    unmatched_source_counter: Counter[str],
    unmatched_target_counter: Counter[str],
    per_lang_counts: dict[str, Counter[str]],
) -> dict[str, Any]:
    total_rows = int(len(enriched_df))

    stats: dict[str, Any] = {
        "n_rows": total_rows,
        "n_exact_pair": int(enriched_df["cognet_exact_pair"].sum()),
        "n_shared_concept": int(enriched_df["cognet_shared_concept"].sum()),
        "n_source_found": int(enriched_df["cognet_source_found"].sum()),
        "n_target_found": int(enriched_df["cognet_target_found"].sum()),
        "n_source_has_alternative_english": int(enriched_df["cognet_source_has_alternative_english"].sum()),
        "avg_source_alternative_english_count": (
            float(enriched_df["cognet_source_alternative_english_count"].mean()) if total_rows else 0.0
        ),
        "n_row_has_any_excluded_word": int(enriched_df["cognet_row_has_any_excluded_word"].sum()),
        "n_context_has_excluded_word": (
            int(enriched_df["L1_context_has_excluded_word"].sum())
            if "L1_context_has_excluded_word" in enriched_df.columns
            else 0
        ),
        "n_no_match": int((enriched_df["cognet_match_type"] == "no_match").sum()),
        "n_unknown_source_language": int((enriched_df["cognet_match_type"] == "unknown_source_language").sum()),
        "exact_pair_rate": float(enriched_df["cognet_exact_pair"].mean()) if total_rows else 0.0,
        "shared_concept_rate": float(enriched_df["cognet_shared_concept"].mean()) if total_rows else 0.0,
        "source_found_rate": float(enriched_df["cognet_source_found"].mean()) if total_rows else 0.0,
        "target_found_rate": float(enriched_df["cognet_target_found"].mean()) if total_rows else 0.0,
        "source_has_alternative_english_rate": (
            float(enriched_df["cognet_source_has_alternative_english"].mean()) if total_rows else 0.0
        ),
        "row_has_any_excluded_word_rate": (
            float(enriched_df["cognet_row_has_any_excluded_word"].mean()) if total_rows else 0.0
        ),
        "match_type_counts": {
            str(k): int(v)
            for k, v in enriched_df["cognet_match_type"].value_counts(dropna=False).to_dict().items()
        },
        "coverage_by_l1": {},
        "top_unmatched_source_words": [
            {"word": word, "count": int(count)}
            for word, count in unmatched_source_counter.most_common(50)
        ],
        "top_unmatched_target_words": [
            {"word": word, "count": int(count)}
            for word, count in unmatched_target_counter.most_common(50)
        ],
        "n_detailed_match_rows": int(len(matches_df)),
        "columns_used": {
            "lang_column": lang_column,
            "source_word_column": source_word_column,
            "target_word_column": target_word_column,
        },
    }

    if lang_column in enriched_df.columns:
        grouped = enriched_df.groupby(lang_column, dropna=False)
        for lang_value, subdf in grouped:
            lang_key = str(lang_value)
            stats["coverage_by_l1"][lang_key] = {
                "n_rows": int(len(subdf)),
                "n_exact_pair": int(subdf["cognet_exact_pair"].sum()),
                "n_shared_concept": int(subdf["cognet_shared_concept"].sum()),
                "n_source_found": int(subdf["cognet_source_found"].sum()),
                "n_target_found": int(subdf["cognet_target_found"].sum()),
                "n_source_has_alternative_english": int(subdf["cognet_source_has_alternative_english"].sum()),
                "avg_source_alternative_english_count": float(subdf["cognet_source_alternative_english_count"].mean()),
                "n_row_has_any_excluded_word": int(subdf["cognet_row_has_any_excluded_word"].sum()),
                "match_type_counts": {
                    str(k): int(v)
                    for k, v in subdf["cognet_match_type"].value_counts(dropna=False).to_dict().items()
                },
            }

    stats["match_type_counts_by_lang_counter"] = {
        lang: {k: int(v) for k, v in counter.items()}
        for lang, counter in per_lang_counts.items()
    }

    return stats


def _load_dataframe_from_config(
    *,
    config_path: Path,
    split_name: str,
) -> pd.DataFrame:
    resolved_cfg = load_and_resolve_config(config_path)
    bundle = load_raw_dataset(resolved_cfg)

    if split_name not in bundle:
        raise ValueError(
            f"Split '{split_name}' was not loaded from config. "
            f"Available splits: {sorted(bundle.keys())}"
        )

    df = bundle[split_name]["df"].copy()
    print(
        json.dumps(
            {
                "input_mode": "config",
                "config": str(config_path),
                "split": split_name,
                "n_rows_loaded": int(len(df)),
                "columns": list(df.columns),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return df


def _load_input_dataframe(
    *,
    input_path: Path | None,
    config_path: Path | None,
    split_name: str | None,
) -> pd.DataFrame:
    if config_path is not None:
        if not split_name:
            raise ValueError("--split is required when using --config.")
        return _load_dataframe_from_config(config_path=config_path, split_name=split_name)

    if input_path is not None:
        df = pd.read_csv(input_path)
        print(
            json.dumps(
                {
                    "input_mode": "csv",
                    "input": str(input_path),
                    "n_rows_loaded": int(len(df)),
                    "columns": list(df.columns),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return df

    raise ValueError("Provide either --config and --split, or --input.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich BEA-style feature CSVs with CogNet lexical/concept match features."
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional direct input CSV file. Prefer --config + --split.",
    )
    input_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved/inheriting experiment config path used with load_raw_dataset().",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test"],
        help="Dataset split to load when using --config.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matches-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--cognet-path", type=Path, required=True)
    parser.add_argument("--lang-column", type=str, default="L1")
    parser.add_argument("--source-word-column", type=str, default="L1_source_word")
    parser.add_argument("--target-word-column", type=str, default="en_target_word")
    parser.add_argument("--target-lang-code", type=str, default="eng")
    parser.add_argument("--id-column", type=str, default="item_id")

    args = parser.parse_args()

    if args.config is None and args.input is None:
        parser.error("One of --config or --input must be provided.")
    if args.config is not None and args.split is None:
        parser.error("--split is required when using --config.")

    return args


def main() -> None:
    args = parse_args()

    df = _load_input_dataframe(
        input_path=args.input,
        config_path=args.config,
        split_name=args.split,
    )

    cognet_df = load_cognet_subset(args.cognet_path)
    indexes = build_indexes(cognet_df)

    enriched_df, matches_df, stats = enrich_dataframe_with_cognet(
        df=df,
        cognet_df=cognet_df,
        indexes=indexes,
        lang_column=args.lang_column,
        source_word_column=args.source_word_column,
        target_word_column=args.target_word_column,
        target_lang_code=args.target_lang_code.strip().lower(),
        id_column=args.id_column,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.matches_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    enriched_df.to_csv(args.output, index=False)
    matches_df.to_csv(args.matches_output, index=False)

    with args.stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "config": str(args.config) if args.config else None,
                "split": args.split,
                "input": str(args.input) if args.input else None,
                "cognet_subset": str(args.cognet_path),
                "output": str(args.output),
                "matches_output": str(args.matches_output),
                "stats_output": str(args.stats_output),
                "n_rows": len(df),
                "n_enriched_rows": len(enriched_df),
                "n_match_rows": len(matches_df),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()