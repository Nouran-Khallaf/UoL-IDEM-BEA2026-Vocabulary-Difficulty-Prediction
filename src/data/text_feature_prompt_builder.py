from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import pandas as pd


@dataclass(slots=True)
class PromptFeatureSpec:
    """
    Specification for one engineered feature to be rendered into text.

    Attributes
    ----------
    column_name:
        Feature column in the dataframe.
    alias:
        Human-readable name used in the prompt.
    kind:
        Rendering strategy:
        - "float"
        - "int"
        - "bool"
        - "str"
    precision:
        Decimal precision for float rendering.
    enabled:
        Whether the feature should be included.
    """
    column_name: str
    alias: str | None = None
    kind: str = "float"
    precision: int = 4
    enabled: bool = True


@dataclass(slots=True)
class TextFeaturePromptConfig:
    """
    Configuration for converting mixed text + engineered features into a single
    textual prompt for transformer regression.

    Parameters
    ----------
    text_columns:
        Ordered text columns to include first.
    feature_specs:
        Feature specs to append after the text section.
    section_labels:
        Whether to add explicit section labels.
    text_header:
        Header for the text section.
    feature_header:
        Header for the feature section.
    missing_token:
        Token used when a value is missing.
    pair_separator:
        Separator between feature-value pairs.
    field_separator:
        Separator between text fields.
    """
    text_columns: list[str]
    feature_specs: list[PromptFeatureSpec]
    section_labels: bool = True
    text_header: str = "TEXT"
    feature_header: str = "FEATURES"
    missing_token: str = "NA"
    pair_separator: str = " | "
    field_separator: str = " [SEP] "


_VALID_KINDS = {"float", "int", "bool", "str"}


def _safe_text(value: object, missing_token: str) -> str:
    if pd.isna(value):
        return missing_token
    text = str(value).strip()
    return text if text else missing_token


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _format_feature_value(
    value: object,
    *,
    kind: str,
    precision: int,
    missing_token: str,
) -> str:
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unsupported feature kind '{kind}'. Expected one of: {sorted(_VALID_KINDS)}")

    if kind == "float":
        x = _safe_float(value)
        if x is None:
            return missing_token
        return f"{x:.{precision}f}"

    if kind == "int":
        x = _safe_float(value)
        if x is None:
            return missing_token
        return str(int(round(x)))

    if kind == "bool":
        x = _safe_float(value)
        if x is None:
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "yes", "1"}:
                    return "true"
                if low in {"false", "no", "0"}:
                    return "false"
                return missing_token
            return missing_token
        return "true" if int(round(x)) != 0 else "false"

    return _safe_text(value, missing_token)


def _validate_config(cfg: TextFeaturePromptConfig) -> None:
    if not cfg.text_columns:
        raise ValueError("text_columns must be a non-empty list.")
    if not cfg.feature_specs:
        raise ValueError("feature_specs must be a non-empty list.")

    for col in cfg.text_columns:
        if not isinstance(col, str) or not col.strip():
            raise ValueError("All text_columns must be non-empty strings.")

    for spec in cfg.feature_specs:
        if not isinstance(spec.column_name, str) or not spec.column_name.strip():
            raise ValueError("Each PromptFeatureSpec.column_name must be a non-empty string.")
        if spec.kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported kind '{spec.kind}' for feature '{spec.column_name}'. "
                f"Expected one of: {sorted(_VALID_KINDS)}"
            )
        if spec.precision < 0:
            raise ValueError("PromptFeatureSpec.precision must be >= 0.")


class TextFeaturePromptBuilder:
    """
    Builder that converts each dataframe row into one prompt string containing:
    - source text fields
    - engineered features rendered as structured text

    This is the core input builder for the third architecture:
    'features appended as text'.
    """

    def __init__(self, cfg: TextFeaturePromptConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def _render_text_section(self, row: pd.Series) -> str:
        parts = [_safe_text(row.get(col), self.cfg.missing_token) for col in self.cfg.text_columns]
        text_body = self.cfg.field_separator.join(parts)

        if not self.cfg.section_labels:
            return text_body

        return f"{self.cfg.text_header}: {text_body}"

    def _render_feature_section(self, row: pd.Series) -> str:
        pairs: list[str] = []

        for spec in self.cfg.feature_specs:
            if not spec.enabled:
                continue

            alias = spec.alias or spec.column_name
            raw_value = row.get(spec.column_name)
            value_text = _format_feature_value(
                raw_value,
                kind=spec.kind,
                precision=spec.precision,
                missing_token=self.cfg.missing_token,
            )
            pairs.append(f"{alias}={value_text}")

        feature_body = self.cfg.pair_separator.join(pairs)

        if not self.cfg.section_labels:
            return feature_body

        return f"{self.cfg.feature_header}: {feature_body}"

    def build_prompt(self, row: pd.Series) -> str:
        text_part = self._render_text_section(row)
        feature_part = self._render_feature_section(row)
        return f"{text_part}{self.cfg.field_separator}{feature_part}".strip()

    def build_prompts(self, df: pd.DataFrame) -> list[str]:
        missing_text = [c for c in self.cfg.text_columns if c not in df.columns]
        if missing_text:
            raise ValueError(f"Missing text columns in dataframe: {missing_text}")

        missing_feature_cols = [
            spec.column_name
            for spec in self.cfg.feature_specs
            if spec.enabled and spec.column_name not in df.columns
        ]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns in dataframe: {missing_feature_cols}")

        prompts: list[str] = []
        for _, row in df.iterrows():
            prompts.append(self.build_prompt(row))
        return prompts


def infer_prompt_feature_specs(
    *,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    exclude_columns: Sequence[str] | None = None,
    default_float_precision: int = 4,
) -> list[PromptFeatureSpec]:
    """
    Infer prompt feature specs from dataframe dtypes.

    Numeric columns become float or int.
    Non-numeric columns become str.

    This is useful as a strong default, after which you can refine the specs
    manually for important features.
    """
    exclude = set(exclude_columns or [])
    specs: list[PromptFeatureSpec] = []

    for col in feature_columns:
        if col in exclude or col not in df.columns:
            continue

        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            kind = "bool"
        elif pd.api.types.is_integer_dtype(series):
            kind = "int"
        elif pd.api.types.is_numeric_dtype(series):
            # detect binary-like numeric features
            uniq = pd.Series(series).dropna().unique()
            if len(uniq) > 0 and set(map(float, uniq)).issubset({0.0, 1.0}):
                kind = "bool"
            else:
                kind = "float"
        else:
            kind = "str"

        specs.append(
            PromptFeatureSpec(
                column_name=col,
                alias=col,
                kind=kind,
                precision=default_float_precision,
                enabled=True,
            )
        )

    return specs