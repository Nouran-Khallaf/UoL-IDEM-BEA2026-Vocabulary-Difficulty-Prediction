from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import spacy
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "spaCy is required for semantic precomputation. Install it via requirements.txt."
    ) from e


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass(slots=True)
class SemanticPrecomputeConfig:
    input_path: Path
    output_path: Path
    split_name: str

    source_lang: str
    target_lang: str

    source_word_column: str
    target_word_column: str

    source_context_column: str | None
    target_context_column: str | None

    source_usas_tags_column: str
    target_usas_tags_column: str
    source_domain_column: str
    target_domain_column: str
    source_domain_score_column: str
    target_domain_score_column: str

    source_lemma_column: str | None
    target_lemma_column: str | None
    add_source_lemma: bool
    add_target_lemma: bool

    overwrite_existing: bool
    keep_all_original_columns: bool


DEFAULT_SOURCE_WORD_COLUMN = "L1_source_word"
DEFAULT_TARGET_WORD_COLUMN = "en_target_word"
DEFAULT_SOURCE_CONTEXT_COLUMN = "L1_context"
DEFAULT_TARGET_CONTEXT_COLUMN = None

DEFAULT_SOURCE_USAS_TAGS_COLUMN = "usas_source_tags"
DEFAULT_TARGET_USAS_TAGS_COLUMN = "usas_target_tags"
DEFAULT_SOURCE_DOMAIN_COLUMN = "source_domain"
DEFAULT_TARGET_DOMAIN_COLUMN = "target_domain"
DEFAULT_SOURCE_DOMAIN_SCORE_COLUMN = "source_domain_score"
DEFAULT_TARGET_DOMAIN_SCORE_COLUMN = "target_domain_score"

DEFAULT_SOURCE_LEMMA_COLUMN = "L1_source_lemma"
DEFAULT_TARGET_LEMMA_COLUMN = "en_target_lemma"


SUPPORTED_LANGS_FOR_PYMUSAS = {
    "en": {
        "spacy_core": "en_core_web_sm",
        "pymusas_model": "en_dual_none_contextual_none",
    },
    "es": {
        "spacy_core": "es_core_news_sm",
        "pymusas_model": "es_dual_upos2usas_contextual_none",
    },
     "cmn": {
        "spacy_core": "zh_core_web_sm",
        "pymusas_model": "cmn_dual_upos2usas_contextual_none",
    },
    "de": {
        "spacy_core": "nl_core_news_sm",
        "pymusas_model": "nl_single_upos2usas_contextual_none",
    },

}

SUPPORTED_LANGS_FOR_LEMMA = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "cmn": "zh_core_web_sm",
}


# -------------------------------------------------
# CLI
# -------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute semantic USAS/domain columns for BEA feature files."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--split-name", type=str, default="unknown")

    parser.add_argument(
        "--source-lang",
        type=str,
        required=True,
        help="Source language code, e.g. de, es, zh.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        help="Target language code, default: en.",
    )

    parser.add_argument("--source-word-column", type=str, default=DEFAULT_SOURCE_WORD_COLUMN)
    parser.add_argument("--target-word-column", type=str, default=DEFAULT_TARGET_WORD_COLUMN)
    parser.add_argument("--source-context-column", type=str, default=DEFAULT_SOURCE_CONTEXT_COLUMN)
    parser.add_argument("--target-context-column", type=str, default=DEFAULT_TARGET_CONTEXT_COLUMN)

    parser.add_argument("--source-usas-tags-column", type=str, default=DEFAULT_SOURCE_USAS_TAGS_COLUMN)
    parser.add_argument("--target-usas-tags-column", type=str, default=DEFAULT_TARGET_USAS_TAGS_COLUMN)
    parser.add_argument("--source-domain-column", type=str, default=DEFAULT_SOURCE_DOMAIN_COLUMN)
    parser.add_argument("--target-domain-column", type=str, default=DEFAULT_TARGET_DOMAIN_COLUMN)
    parser.add_argument("--source-domain-score-column", type=str, default=DEFAULT_SOURCE_DOMAIN_SCORE_COLUMN)
    parser.add_argument("--target-domain-score-column", type=str, default=DEFAULT_TARGET_DOMAIN_SCORE_COLUMN)

    parser.add_argument("--source-lemma-column", type=str, default=DEFAULT_SOURCE_LEMMA_COLUMN)
    parser.add_argument("--target-lemma-column", type=str, default=DEFAULT_TARGET_LEMMA_COLUMN)

    parser.add_argument(
        "--add-source-lemma",
        action="store_true",
        help="Add a source lemma column if possible.",
    )
    parser.add_argument(
        "--add-target-lemma",
        action="store_true",
        help="Add a target lemma column if possible.",
    )

    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite semantic columns if they already exist.",
    )
    parser.add_argument(
        "--drop-extra-generated-columns",
        action="store_true",
        help="If set, keep only original columns + semantic columns. Default keeps all.",
    )

    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> SemanticPrecomputeConfig:
    return SemanticPrecomputeConfig(
        input_path=args.input,
        output_path=args.output,
        split_name=str(args.split_name),
        source_lang=str(args.source_lang).strip().lower(),
        target_lang=str(args.target_lang).strip().lower(),
        source_word_column=str(args.source_word_column),
        target_word_column=str(args.target_word_column),
        source_context_column=str(args.source_context_column) if args.source_context_column else None,
        target_context_column=str(args.target_context_column) if args.target_context_column else None,
        source_usas_tags_column=str(args.source_usas_tags_column),
        target_usas_tags_column=str(args.target_usas_tags_column),
        source_domain_column=str(args.source_domain_column),
        target_domain_column=str(args.target_domain_column),
        source_domain_score_column=str(args.source_domain_score_column),
        target_domain_score_column=str(args.target_domain_score_column),
        source_lemma_column=str(args.source_lemma_column) if args.source_lemma_column else None,
        target_lemma_column=str(args.target_lemma_column) if args.target_lemma_column else None,
        add_source_lemma=bool(args.add_source_lemma),
        add_target_lemma=bool(args.add_target_lemma),
        overwrite_existing=bool(args.overwrite_existing),
        keep_all_original_columns=not bool(args.drop_extra_generated_columns),
    )


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _stringify_list(values: list[str]) -> str:
    if not values:
        return ""
    return "|".join(values)


def _usas_to_coarse_domain(tag: str) -> str:
    tag = _normalized_text(tag).lower()
    if not tag:
        return ""
    m = re.match(r"([a-z]+)", tag)
    if m:
        return m.group(1)
    return tag[:1] if tag else ""


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def _should_skip_existing(df: pd.DataFrame, cols: list[str], overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return False
    present = [c for c in cols if c in df.columns]
    if len(present) != len(cols):
        return False
    return all(df[c].notna().any() for c in present)


# -------------------------------------------------
# PyMUSAS loading
# -------------------------------------------------
def build_pymusas_pipeline(spacy_core_model: str, pymusas_model: str):
    """
    Build a lightweight pipeline = spaCy core model + PyMUSAS rule-based tagger.
    """
    nlp = spacy.load(spacy_core_model, exclude=["parser", "ner"])
    tagger_pipe = spacy.load(pymusas_model)
    nlp.add_pipe("pymusas_rule_based_tagger", source=tagger_pipe)
    return nlp


def maybe_build_lang_pipeline(lang: str):
    """
    Return a spaCy+PyMUSAS pipeline if available for this language, else None.
    """
    spec = SUPPORTED_LANGS_FOR_PYMUSAS.get(lang)
    if spec is None:
        return None
    pymusas_model = spec.get("pymusas_model")
    if not pymusas_model:
        return None
    return build_pymusas_pipeline(spec["spacy_core"], pymusas_model)


# -------------------------------------------------
# Lemma loading
# -------------------------------------------------
def build_lemmatizer_pipeline(spacy_core_model: str):
    """
    Build a lightweight spaCy lemmatization pipeline.
    """
    return spacy.load(spacy_core_model, exclude=["parser", "ner"])


def maybe_build_lemmatizer(lang: str):
    model_name = SUPPORTED_LANGS_FOR_LEMMA.get(lang)
    if model_name is None:
        return None
    try:
        return build_lemmatizer_pipeline(model_name)
    except Exception:
        return None


# -------------------------------------------------
# Token tagging / token matching
# -------------------------------------------------
def _pick_best_token(doc, word: str):
    """
    Pick the token most likely corresponding to the inserted word.
    """
    word_norm = _normalized_text(word).lower()
    if not word_norm or len(doc) == 0:
        return None

    for token in doc:
        if token.text.strip().lower() == word_norm:
            return token

    best_token = None
    best_score = -1.0
    for token in doc:
        tok = token.text.strip().lower()
        if not tok:
            continue
        char_overlap = len(set(tok) & set(word_norm)) / max(1, len(set(tok) | set(word_norm)))
        starts = float(tok.startswith(word_norm[:2])) if len(word_norm) >= 2 else 0.0
        score = 0.8 * char_overlap + 0.2 * starts
        if score > best_score:
            best_score = score
            best_token = token
    return best_token


def _build_template_text(word: str, context: str | None, lang: str) -> str:
    """
    Put the word in a tiny context to reduce tagging ambiguity.
    """
    word = _normalized_text(word)
    ctx = _normalized_text(context)

    if ctx:
        return f"{ctx} {word}"

    if lang == "en":
        return f"This is {word}."
    if lang == "es":
        return f"Esto es {word}."
    if lang == "de":
        return f"Das ist {word}."
    return word


def word_to_usas_tags(word: Any, nlp, lang: str, context: Any = None) -> list[str]:
    if nlp is None:
        return []

    word_text = _normalized_text(word)
    if not word_text:
        return []

    text = _build_template_text(word_text, context, lang)
    try:
        doc = nlp(text)
    except Exception:
        return []

    token = _pick_best_token(doc, word_text)
    if token is None:
        return []

    tags = []
    try:
        raw_tags = token._.pymusas_tags
    except Exception:
        raw_tags = []

    for t in raw_tags or []:
        tt = _normalized_text(t)
        if tt:
            tags.append(tt)

    seen = set()
    uniq = []
    for tag in tags:
        if tag not in seen:
            uniq.append(tag)
            seen.add(tag)
    return uniq


def word_to_lemma(word: Any, nlp, lang: str, context: Any = None) -> str:
    if nlp is None:
        return _normalized_text(word).lower()

    word_text = _normalized_text(word)
    if not word_text:
        return ""

    text = _build_template_text(word_text, context, lang)
    try:
        doc = nlp(text)
    except Exception:
        return word_text.lower()

    token = _pick_best_token(doc, word_text)
    if token is None:
        return word_text.lower()

    lemma = _normalized_text(getattr(token, "lemma_", ""))
    if not lemma:
        return word_text.lower()

    return lemma.lower()


def tags_to_domains(tags: list[str]) -> list[str]:
    domains: list[str] = []
    seen = set()
    for tag in tags:
        dom = _usas_to_coarse_domain(tag)
        if dom and dom not in seen:
            domains.append(dom)
            seen.add(dom)
    return domains


def default_scores_for_items(items: list[str]) -> list[float]:
    if not items:
        return []
    return [1.0] * len(items)


# -------------------------------------------------
# Row-level semantic annotation
# -------------------------------------------------
def annotate_side(
    series_word: pd.Series,
    *,
    series_context: pd.Series | None,
    lang: str,
    nlp,
) -> tuple[list[str], list[str], list[str]]:
    tag_strings: list[str] = []
    domain_strings: list[str] = []
    score_strings: list[str] = []

    if series_context is None:
        series_context = pd.Series([None] * len(series_word), index=series_word.index)

    for word, context in zip(series_word, series_context):
        tags = word_to_usas_tags(word, nlp, lang, context)
        domains = tags_to_domains(tags)
        scores = default_scores_for_items(domains)

        tag_strings.append(_stringify_list(tags))
        domain_strings.append(_stringify_list(domains))
        score_strings.append(_stringify_list([str(float(x)) for x in scores]))

    return tag_strings, domain_strings, score_strings


def annotate_lemmas(
    series_word: pd.Series,
    *,
    series_context: pd.Series | None,
    lang: str,
    nlp,
) -> list[str]:
    lemma_strings: list[str] = []

    if series_context is None:
        series_context = pd.Series([None] * len(series_word), index=series_word.index)

    for word, context in zip(series_word, series_context):
        lemma_strings.append(word_to_lemma(word, nlp, lang, context))

    return lemma_strings


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def precompute_semantic_columns(df: pd.DataFrame, cfg: SemanticPrecomputeConfig) -> pd.DataFrame:
    result = df.copy()

    required = [cfg.source_word_column, cfg.target_word_column]
    missing_required = [c for c in required if c not in result.columns]
    if missing_required:
        raise ValueError(
            f"Missing required input columns for semantic precomputation: {missing_required}"
        )

    target_cols = [
        cfg.source_usas_tags_column,
        cfg.target_usas_tags_column,
        cfg.source_domain_column,
        cfg.target_domain_column,
        cfg.source_domain_score_column,
        cfg.target_domain_score_column,
    ]

    if _should_skip_existing(result, target_cols, cfg.overwrite_existing):
        semantic_already_present = True
    else:
        semantic_already_present = False

    source_nlp = maybe_build_lang_pipeline(cfg.source_lang)
    target_nlp = maybe_build_lang_pipeline(cfg.target_lang)

    source_lemma_nlp = maybe_build_lemmatizer(cfg.source_lang) if cfg.add_source_lemma else None
    target_lemma_nlp = maybe_build_lemmatizer(cfg.target_lang) if cfg.add_target_lemma else None

    source_context_series = (
        result[cfg.source_context_column]
        if cfg.source_context_column and cfg.source_context_column in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )
    target_context_series = (
        result[cfg.target_context_column]
        if cfg.target_context_column and cfg.target_context_column in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )

    if not semantic_already_present:
        tgt_tags, tgt_domains, tgt_scores = annotate_side(
            result[cfg.target_word_column],
            series_context=target_context_series,
            lang=cfg.target_lang,
            nlp=target_nlp,
        )

        if source_nlp is not None:
            src_tags, src_domains, src_scores = annotate_side(
                result[cfg.source_word_column],
                series_context=source_context_series,
                lang=cfg.source_lang,
                nlp=source_nlp,
            )
        else:
            src_tags = [""] * len(result)
            src_domains = [""] * len(result)
            src_scores = [""] * len(result)

        result[cfg.source_usas_tags_column] = src_tags
        result[cfg.target_usas_tags_column] = tgt_tags
        result[cfg.source_domain_column] = src_domains
        result[cfg.target_domain_column] = tgt_domains
        result[cfg.source_domain_score_column] = src_scores
        result[cfg.target_domain_score_column] = tgt_scores

    if cfg.add_target_lemma and cfg.target_lemma_column:
        if cfg.overwrite_existing or cfg.target_lemma_column not in result.columns:
            result[cfg.target_lemma_column] = annotate_lemmas(
                result[cfg.target_word_column],
                series_context=target_context_series,
                lang=cfg.target_lang,
                nlp=target_lemma_nlp,
            )

    if cfg.add_source_lemma and cfg.source_lemma_column:
        if cfg.overwrite_existing or cfg.source_lemma_column not in result.columns:
            result[cfg.source_lemma_column] = annotate_lemmas(
                result[cfg.source_word_column],
                series_context=source_context_series,
                lang=cfg.source_lang,
                nlp=source_lemma_nlp,
            )

    result["semantic_source_lang_supported"] = int(source_nlp is not None)
    result["semantic_target_lang_supported"] = int(target_nlp is not None)
    result["lemma_source_lang_supported"] = int(source_lemma_nlp is not None) if cfg.add_source_lemma else 0
    result["lemma_target_lang_supported"] = int(target_lemma_nlp is not None) if cfg.add_target_lemma else 0

    return result


# -------------------------------------------------
# Reporting
# -------------------------------------------------
def build_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, cfg: SemanticPrecomputeConfig) -> dict[str, Any]:
    def nonempty_rate(col: str) -> float:
        if col not in df_after.columns:
            return 0.0
        vals = df_after[col].fillna("").astype(str).str.strip()
        return float((vals != "").mean())

    lemma_columns_added = []
    if cfg.add_source_lemma and cfg.source_lemma_column:
        lemma_columns_added.append(cfg.source_lemma_column)
    if cfg.add_target_lemma and cfg.target_lemma_column:
        lemma_columns_added.append(cfg.target_lemma_column)

    summary = {
        "split_name": cfg.split_name,
        "n_rows": int(len(df_after)),
        "source_lang": cfg.source_lang,
        "target_lang": cfg.target_lang,
        "source_lang_supported_by_pymusas": bool(
            cfg.source_lang in SUPPORTED_LANGS_FOR_PYMUSAS
            and SUPPORTED_LANGS_FOR_PYMUSAS[cfg.source_lang].get("pymusas_model")
        ),
        "target_lang_supported_by_pymusas": bool(
            cfg.target_lang in SUPPORTED_LANGS_FOR_PYMUSAS
            and SUPPORTED_LANGS_FOR_PYMUSAS[cfg.target_lang].get("pymusas_model")
        ),
        "source_lang_supported_by_lemmatizer": bool(cfg.source_lang in SUPPORTED_LANGS_FOR_LEMMA),
        "target_lang_supported_by_lemmatizer": bool(cfg.target_lang in SUPPORTED_LANGS_FOR_LEMMA),
        "columns_added": [
            cfg.source_usas_tags_column,
            cfg.target_usas_tags_column,
            cfg.source_domain_column,
            cfg.target_domain_column,
            cfg.source_domain_score_column,
            cfg.target_domain_score_column,
        ],
        "lemma_columns_added": lemma_columns_added,
        "nonempty_rate": {
            cfg.source_usas_tags_column: nonempty_rate(cfg.source_usas_tags_column),
            cfg.target_usas_tags_column: nonempty_rate(cfg.target_usas_tags_column),
            cfg.source_domain_column: nonempty_rate(cfg.source_domain_column),
            cfg.target_domain_column: nonempty_rate(cfg.target_domain_column),
            **(
                {cfg.source_lemma_column: nonempty_rate(cfg.source_lemma_column)}
                if cfg.add_source_lemma and cfg.source_lemma_column
                else {}
            ),
            **(
                {cfg.target_lemma_column: nonempty_rate(cfg.target_lemma_column)}
                if cfg.add_target_lemma and cfg.target_lemma_column
                else {}
            ),
        },
    }
    return summary


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_path}")

    df = pd.read_csv(cfg.input_path)
    df_before = df.copy()

    df_out = precompute_semantic_columns(df, cfg)

    if not cfg.keep_all_original_columns:
        keep_cols = list(df_before.columns) + [
            cfg.source_usas_tags_column,
            cfg.target_usas_tags_column,
            cfg.source_domain_column,
            cfg.target_domain_column,
            cfg.source_domain_score_column,
            cfg.target_domain_score_column,
            "semantic_source_lang_supported",
            "semantic_target_lang_supported",
            "lemma_source_lang_supported",
            "lemma_target_lang_supported",
        ]
        if cfg.add_source_lemma and cfg.source_lemma_column:
            keep_cols.append(cfg.source_lemma_column)
        if cfg.add_target_lemma and cfg.target_lemma_column:
            keep_cols.append(cfg.target_lemma_column)

        keep_cols = [c for c in keep_cols if c in df_out.columns]
        df_out = df_out[keep_cols]

    _ensure_parent_dir(cfg.output_path)
    df_out.to_csv(cfg.output_path, index=False)

    summary = build_summary(df_before, df_out, cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()