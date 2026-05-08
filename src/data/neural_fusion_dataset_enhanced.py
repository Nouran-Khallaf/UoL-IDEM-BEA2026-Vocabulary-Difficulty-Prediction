from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoTokenizer


_VALID_TEXT_JOIN_MODES = {"sep", "newline", "space"}


@dataclass(slots=True)
class NeuralFusionDatasetConfig:
    """
    Configuration for transformer + tabular regression dataset construction.
    """
    encoder_name: str
    text_columns: list[str]
    numeric_feature_columns: list[str]
    target_column: str
    max_length: int = 128
    text_join_mode: str = "sep"
    add_special_tokens: bool = True


def _validate_join_mode(mode: str) -> str:
    resolved = mode.strip().lower()
    if resolved not in _VALID_TEXT_JOIN_MODES:
        raise ValueError(
            f"Unsupported text_join_mode '{mode}'. "
            f"Expected one of: {sorted(_VALID_TEXT_JOIN_MODES)}"
        )
    return resolved


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


class TabularFeaturePreprocessor:
    """
    numeric -> median imputation -> standard scaling
    """

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_columns_: list[str] | None = None
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> "TabularFeaturePreprocessor":
        feature_columns = [str(c).strip() for c in feature_columns if str(c).strip()]
        if not feature_columns:
            raise ValueError("feature_columns must be a non-empty list.")

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing numeric feature columns in dataframe: {missing}")

        X = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X_imp = self.imputer.fit_transform(X)
        self.scaler.fit(X_imp)

        self.feature_columns_ = list(feature_columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_ or self.feature_columns_ is None:
            raise ValueError("TabularFeaturePreprocessor must be fitted before transform().")

        missing = [c for c in self.feature_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing numeric feature columns in dataframe: {missing}")

        X = df[self.feature_columns_].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
        self.fit(df, feature_columns)
        return self.transform(df)

    @property
    def n_features_out_(self) -> int:
        if self.feature_columns_ is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return len(self.feature_columns_)


class NeuralFusionRegressionDataset(Dataset):
    def __init__(
        self,
        *,
        encodings: dict[str, torch.Tensor],
        tabular_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if "input_ids" not in encodings or "attention_mask" not in encodings:
            raise ValueError("encodings must contain 'input_ids' and 'attention_mask'.")

        if len(tabular_matrix) != len(labels):
            raise ValueError(
                f"tabular_matrix and labels length mismatch: {len(tabular_matrix)} != {len(labels)}"
            )

        if encodings["input_ids"].shape[0] != len(labels):
            raise ValueError(
                "Tokenizer outputs and labels length mismatch: "
                f"{encodings['input_ids'].shape[0]} != {len(labels)}"
            )

        self.encodings = encodings
        self.tabular_features = torch.tensor(tabular_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "tabular_features": self.tabular_features[idx],
            "labels": self.labels[idx],
        }


class NeuralFusionDatasetBuilder:
    """
    Builder for train/dev neural fusion datasets.
    """

    def __init__(self, cfg: NeuralFusionDatasetConfig) -> None:
        if not cfg.encoder_name or not cfg.encoder_name.strip():
            raise ValueError("encoder_name must be a non-empty string.")
        if not cfg.text_columns:
            raise ValueError("text_columns must be a non-empty list.")
        if not cfg.numeric_feature_columns:
            raise ValueError("numeric_feature_columns must be a non-empty list.")
        if not cfg.target_column or not cfg.target_column.strip():
            raise ValueError("target_column must be a non-empty string.")
        if cfg.max_length <= 0:
            raise ValueError("max_length must be > 0.")

        self.cfg = NeuralFusionDatasetConfig(
            encoder_name=cfg.encoder_name,
            text_columns=list(cfg.text_columns),
            numeric_feature_columns=list(cfg.numeric_feature_columns),
            target_column=cfg.target_column,
            max_length=int(cfg.max_length),
            text_join_mode=_validate_join_mode(cfg.text_join_mode),
            add_special_tokens=bool(cfg.add_special_tokens),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.encoder_name)
        self.tabular_preprocessor = TabularFeaturePreprocessor()

    def _join_text_parts(self, parts: Sequence[str]) -> str:
        if self.cfg.text_join_mode == "sep":
            sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else "[SEP]"
            return f" {sep_token} ".join(parts)
        if self.cfg.text_join_mode == "newline":
            return "\n".join(parts)
        if self.cfg.text_join_mode == "space":
            return " ".join(parts)
        raise ValueError(f"Unsupported join mode '{self.cfg.text_join_mode}'.")

    def _build_texts(self, df: pd.DataFrame) -> list[str]:
        missing = [c for c in self.cfg.text_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing text columns in dataframe: {missing}")

        texts: list[str] = []
        for _, row in df.iterrows():
            parts = [_safe_text(row[col]) for col in self.cfg.text_columns]
            text = self._join_text_parts(parts).strip()
            texts.append(text)
        return texts

    def fit_tabular_preprocessor(self, train_df: pd.DataFrame) -> None:
        self.tabular_preprocessor.fit(train_df, self.cfg.numeric_feature_columns)

    def build_encodings(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        texts = self._build_texts(df)

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=self.cfg.add_special_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return encodings

    def build_tabular_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return self.tabular_preprocessor.transform(df)

    def build_labels(self, df: pd.DataFrame) -> np.ndarray:
        if self.cfg.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.cfg.target_column}' not found in dataframe."
            )

        y = pd.to_numeric(df[self.cfg.target_column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError(
                f"Target column '{self.cfg.target_column}' contains NaN after numeric conversion."
            )
        return y.astype(np.float32)

    def build_dataset(self, df: pd.DataFrame) -> NeuralFusionRegressionDataset:
        encodings = self.build_encodings(df)
        tabular_matrix = self.build_tabular_matrix(df)
        labels = self.build_labels(df)

        return NeuralFusionRegressionDataset(
            encodings=encodings,
            tabular_matrix=tabular_matrix,
            labels=labels,
        )