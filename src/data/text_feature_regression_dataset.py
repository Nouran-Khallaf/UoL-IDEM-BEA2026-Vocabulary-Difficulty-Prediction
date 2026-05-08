from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.text_feature_prompt_builder import (
    PromptFeatureSpec,
    TextFeaturePromptBuilder,
    TextFeaturePromptConfig,
    infer_prompt_feature_specs,
)


@dataclass(slots=True)
class TextFeatureRegressionDatasetConfig:
    """
    Configuration for the prompt-based regression dataset.

    Parameters
    ----------
    encoder_name:
        Hugging Face encoder name used for tokenization.
    text_columns:
        Original text columns to place in the prompt text section.
    feature_columns:
        Engineered feature columns to verbalize into the prompt.
    target_column:
        Gold regression target.
    max_length:
        Maximum tokenizer sequence length.
    add_special_tokens:
        Whether tokenizer special tokens should be added.
    prompt_section_labels:
        Whether to include explicit section headers such as TEXT / FEATURES.
    text_header:
        Header string for the text block.
    feature_header:
        Header string for the feature block.
    missing_token:
        Placeholder used when a value is missing.
    pair_separator:
        Separator between rendered feature-value pairs.
    field_separator:
        Separator between text fields and prompt sections.
    """
    encoder_name: str
    text_columns: list[str]
    feature_columns: list[str]
    target_column: str
    max_length: int = 192
    add_special_tokens: bool = True
    prompt_section_labels: bool = True
    text_header: str = "TEXT"
    feature_header: str = "FEATURES"
    missing_token: str = "NA"
    pair_separator: str = " | "
    field_separator: str = " [SEP] "
    default_float_precision: int = 4


def _validate_cfg(cfg: TextFeatureRegressionDatasetConfig) -> None:
    if not cfg.encoder_name or not cfg.encoder_name.strip():
        raise ValueError("encoder_name must be a non-empty string.")
    if not cfg.text_columns:
        raise ValueError("text_columns must be a non-empty list.")
    if not cfg.feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    if not cfg.target_column or not cfg.target_column.strip():
        raise ValueError("target_column must be a non-empty string.")
    if cfg.max_length <= 0:
        raise ValueError("max_length must be > 0.")
    if cfg.default_float_precision < 0:
        raise ValueError("default_float_precision must be >= 0.")


class PromptRegressionDataset(Dataset):
    """
    Tokenized prompt dataset for text-only regression.

    Each item returns:
    - input_ids
    - attention_mask
    - labels
    """

    def __init__(
        self,
        *,
        encodings: dict[str, torch.Tensor],
        labels: np.ndarray,
        prompts: list[str] | None = None,
    ) -> None:
        if "input_ids" not in encodings or "attention_mask" not in encodings:
            raise ValueError("encodings must contain 'input_ids' and 'attention_mask'.")

        n_rows = int(encodings["input_ids"].shape[0])
        if n_rows != len(labels):
            raise ValueError(
                f"Tokenizer output / labels mismatch: {n_rows} != {len(labels)}"
            )

        if prompts is not None and len(prompts) != len(labels):
            raise ValueError(
                f"Prompt / labels mismatch: {len(prompts)} != {len(labels)}"
            )

        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class TextFeatureRegressionDatasetBuilder:
    """
    Builder for the prompt-based regression dataset.

    This is the main dataset builder for the third architecture:
    engineered features appended as structured text.
    """

    def __init__(
        self,
        cfg: TextFeatureRegressionDatasetConfig,
        *,
        feature_specs: list[PromptFeatureSpec] | None = None,
    ) -> None:
        _validate_cfg(cfg)
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)

        if feature_specs is None:
            self.feature_specs_: list[PromptFeatureSpec] | None = None
        else:
            self.feature_specs_ = list(feature_specs)

    def _build_prompt_config(
        self,
        df: pd.DataFrame,
    ) -> TextFeaturePromptConfig:
        if self.feature_specs_ is None:
            self.feature_specs_ = infer_prompt_feature_specs(
                df=df,
                feature_columns=self.cfg.feature_columns,
                exclude_columns=[self.cfg.target_column, "item_id", *self.cfg.text_columns],
                default_float_precision=self.cfg.default_float_precision,
            )

        return TextFeaturePromptConfig(
            text_columns=self.cfg.text_columns,
            feature_specs=self.feature_specs_,
            section_labels=self.cfg.prompt_section_labels,
            text_header=self.cfg.text_header,
            feature_header=self.cfg.feature_header,
            missing_token=self.cfg.missing_token,
            pair_separator=self.cfg.pair_separator,
            field_separator=self.cfg.field_separator,
        )

    def build_prompts(self, df: pd.DataFrame) -> list[str]:
        prompt_cfg = self._build_prompt_config(df)
        builder = TextFeaturePromptBuilder(prompt_cfg)
        return builder.build_prompts(df)

    def build_encodings(self, df: pd.DataFrame) -> tuple[dict[str, torch.Tensor], list[str]]:
        prompts = self.build_prompts(df)

        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=self.cfg.add_special_tokens,
            return_tensors="pt",
        )
        return encodings, prompts

    def build_labels(self, df: pd.DataFrame) -> np.ndarray:
        if self.cfg.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.cfg.target_column}' not found in dataframe."
            )

        labels = pd.to_numeric(df[self.cfg.target_column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(labels).any():
            raise ValueError(
                f"Target column '{self.cfg.target_column}' contains NaN after numeric conversion."
            )

        return labels.astype(np.float32)

    def build_dataset(
        self,
        df: pd.DataFrame,
        *,
        return_prompts: bool = False,
    ) -> PromptRegressionDataset | tuple[PromptRegressionDataset, list[str]]:
        encodings, prompts = self.build_encodings(df)
        labels = self.build_labels(df)

        dataset = PromptRegressionDataset(
            encodings=encodings,
            labels=labels,
            prompts=prompts if return_prompts else None,
        )

        if return_prompts:
            return dataset, prompts
        return dataset

    def get_feature_specs(self) -> list[PromptFeatureSpec]:
        if self.feature_specs_ is None:
            raise ValueError(
                "Feature specs have not been initialized yet. "
                "Call build_prompts/build_dataset first, or pass feature_specs explicitly."
            )
        return list(self.feature_specs_)


def infer_default_prompt_feature_columns(
    *,
    df: pd.DataFrame,
    exclude_columns: Sequence[str] | None = None,
) -> list[str]:
    """
    Infer a default set of engineered feature columns to verbalize.

    By default, exclude:
    - item_id
    - target columns if passed in exclude_columns
    - obvious raw text fields if passed in exclude_columns

    This helper is intentionally conservative and returns all remaining columns.
    """
    exclude = set(exclude_columns or [])
    feature_columns: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        feature_columns.append(col)

    return feature_columns