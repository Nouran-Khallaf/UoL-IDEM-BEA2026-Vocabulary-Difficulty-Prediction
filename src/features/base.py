from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import ExperimentRuntimeError, ModelConfigurationError


@dataclass
class ModelArtifacts:
    """
    Container for model outputs and optional saved artifacts.
    """
    feature_names: list[str] = field(default_factory=list)
    coefficients: dict[str, float] | None = None
    intercept: float | None = None
    feature_importance: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModelRunner(ABC):
    """
    Abstract base class for all model runners.
    """

    model_name: str = "base"
    model_family: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ModelConfigurationError(
                "Model config must be a dictionary.",
                context={"received_type": type(config).__name__},
            )
        self.config = config
        self.is_fitted: bool = False
        self.model: Any = None
        self.artifacts = ModelArtifacts()

    @abstractmethod
    def build_model(self) -> Any:
        """
        Construct the underlying model object.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseModelRunner":
        """
        Fit the model on training data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Predict on new data.
        """
        raise NotImplementedError

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_family(self) -> str:
        return self.model_family

    def get_artifacts(self) -> ModelArtifacts:
        return self.artifacts

    def validate_X(self, X, *, allow_dataframe: bool = True) -> np.ndarray:
        """
        Validate and normalize feature input.
        Returns a 2D numpy array.
        """
        if allow_dataframe and isinstance(X, pd.DataFrame):
            if X.empty:
                raise ExperimentRuntimeError("Input feature DataFrame is empty.")
            try:
                values = X.to_numpy(dtype=float)
            except Exception as e:
                raise ExperimentRuntimeError(
                    "Failed to convert DataFrame X to numeric numpy array.",
                    context={"error": str(e)},
                ) from e
            self.artifacts.feature_names = list(X.columns)
        else:
            try:
                values = np.asarray(X, dtype=float)
            except Exception as e:
                raise ExperimentRuntimeError(
                    "Failed to convert X to numeric numpy array.",
                    context={"error": str(e)},
                ) from e

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if values.ndim != 2:
            raise ExperimentRuntimeError(
                "Model input X must be a 2D matrix.",
                context={"ndim": int(values.ndim)},
            )

        if values.shape[0] == 0:
            raise ExperimentRuntimeError("Model input X has zero rows.")

        if values.shape[1] == 0:
            raise ExperimentRuntimeError("Model input X has zero columns.")

        if np.isnan(values).any():
            raise ExperimentRuntimeError(
                "Model input X contains NaN values.",
                context={"nan_count": int(np.isnan(values).sum())},
            )

        if np.isinf(values).any():
            raise ExperimentRuntimeError(
                "Model input X contains infinite values.",
                context={"inf_count": int(np.isinf(values).sum())},
            )

        return values

    def validate_y(self, y) -> np.ndarray:
        """
        Validate and normalize target input.
        Returns a 1D numpy array.
        """
        try:
            values = np.asarray(y, dtype=float)
        except Exception as e:
            raise ExperimentRuntimeError(
                "Failed to convert y to numeric numpy array.",
                context={"error": str(e)},
            ) from e

        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim > 1:
            values = values.reshape(-1)

        if values.shape[0] == 0:
            raise ExperimentRuntimeError("Target y is empty.")

        if np.isnan(values).any():
            raise ExperimentRuntimeError(
                "Target y contains NaN values.",
                context={"nan_count": int(np.isnan(values).sum())},
            )

        if np.isinf(values).any():
            raise ExperimentRuntimeError(
                "Target y contains infinite values.",
                context={"inf_count": int(np.isinf(values).sum())},
            )

        return values

    def validate_X_y(self, X, y, *, allow_dataframe: bool = True) -> tuple[np.ndarray, np.ndarray]:
        X_values = self.validate_X(X, allow_dataframe=allow_dataframe)
        y_values = self.validate_y(y)

        if X_values.shape[0] != y_values.shape[0]:
            raise ExperimentRuntimeError(
                "X and y have incompatible number of rows.",
                context={
                    "n_rows_X": int(X_values.shape[0]),
                    "n_rows_y": int(y_values.shape[0]),
                },
            )

        return X_values, y_values

    def check_is_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise ExperimentRuntimeError(
                "Model has not been fitted yet.",
                context={"model_name": self.model_name},
            )

    def set_feature_names(self, feature_names: list[str]) -> None:
        self.artifacts.feature_names = list(feature_names)

    def get_feature_names(self) -> list[str]:
        return list(self.artifacts.feature_names)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "is_fitted": self.is_fitted,
            "n_features": len(self.artifacts.feature_names),
        }