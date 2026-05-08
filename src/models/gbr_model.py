from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.core.exceptions import ExperimentRuntimeError, ModelConfigurationError
from src.models.base import BaseModelRunner


class GBRRunner(BaseModelRunner):


    model_name = "gbr"
    model_family = "tree_ensemble"

    DEFAULT_CONFIG: dict[str, Any] = {
        "preprocessing": {
            "numeric_imputer": {
                "enabled": True,
                "strategy": "median",
            },
            "categorical_imputer": {
                "enabled": False,
                "strategy": "most_frequent",
            },
            "scaling": {
                "enabled": False,
                "method": "standard",
                "with_mean": True,
                "with_std": True,
            },
            "variance_filter": {
                "enabled": False,
                "threshold": 0.0,
            },
            "correlation_filter": {
                "enabled": False,
                "threshold": 0.95,
                "method": "pearson",
            },
            "clip_extremes": {
                "enabled": False,
                "lower_quantile": 0.001,
                "upper_quantile": 0.999,
            },
        },
        "parameters": {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": 3,
            "min_impurity_decrease": 0.0,
            "init": None,
            "random_state": 42,
            "max_features": None,
            "alpha": 0.9,
            "verbose": 0,
            "max_leaf_nodes": None,
            "warm_start": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": None,
            "tol": 1e-4,
            "ccp_alpha": 0.0,
        },
        "training": {
            "use_sample_weights": False,
        },
        "artifacts": {
            "save_feature_importance": True,
            "save_feature_order": True,
        },
        "failure_policy": {
            "fail_on_nan_after_preprocessing": True,
            "fail_on_inf_after_preprocessing": True,
            "fail_on_empty_feature_matrix": True,
            "warn_on_constant_prediction": True,
        },
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config)
        self.resolved_model_config = self._resolve_model_config(config)

    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, override_value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(override_value, dict)
            ):
                merged[key] = GBRRunner._deep_merge_dicts(merged[key], override_value)
            else:
                merged[key] = deepcopy(override_value)
        return merged

    def _resolve_model_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(cfg, dict):
            raise ModelConfigurationError(
                "Experiment config must be a dictionary.",
                context={"received_type": type(cfg).__name__},
            )

        model_name = cfg.get("model_name")
        if model_name is None:
            raise ModelConfigurationError("Missing required field 'model_name'.")
        if str(model_name).strip().lower() != self.model_name:
            raise ModelConfigurationError(
                "GBRRunner received a config for a different model.",
                context={
                    "expected_model_name": self.model_name,
                    "received_model_name": model_name,
                },
            )

        model_overrides = cfg.get("model_overrides", {})
        if model_overrides is None:
            model_overrides = {}
        if not isinstance(model_overrides, dict):
            raise ModelConfigurationError(
                "'model_overrides' must be a dictionary if provided.",
                context={"received_type": type(model_overrides).__name__},
            )

        resolved = self._deep_merge_dicts(self.DEFAULT_CONFIG, model_overrides)
        self._validate_resolved_model_config(resolved)
        return resolved

    def _validate_resolved_model_config(self, model_cfg: dict[str, Any]) -> None:
        params = model_cfg.get("parameters", {})
        if not isinstance(params, dict):
            raise ModelConfigurationError("'parameters' must be a dictionary.")

        required_numeric_positive = [
            "learning_rate",
            "n_estimators",
        ]
        for key in required_numeric_positive:
            if key not in params:
                raise ModelConfigurationError(
                    f"Missing required GBR parameter '{key}'."
                )

        if float(params["learning_rate"]) <= 0:
            raise ModelConfigurationError(
                "GBR 'learning_rate' must be > 0.",
                context={"learning_rate": params["learning_rate"]},
            )

        if int(params["n_estimators"]) <= 0:
            raise ModelConfigurationError(
                "GBR 'n_estimators' must be > 0.",
                context={"n_estimators": params["n_estimators"]},
            )

        if "subsample" in params:
            subsample = float(params["subsample"])
            if not (0 < subsample <= 1):
                raise ModelConfigurationError(
                    "GBR 'subsample' must be in (0, 1].",
                    context={"subsample": subsample},
                )

        if "max_depth" in params and params["max_depth"] is not None:
            if int(params["max_depth"]) <= 0:
                raise ModelConfigurationError(
                    "GBR 'max_depth' must be > 0 when provided.",
                    context={"max_depth": params["max_depth"]},
                )

    def get_preprocessing_config(self) -> dict[str, Any]:
        return deepcopy(self.resolved_model_config.get("preprocessing", {}))

    def get_training_config(self) -> dict[str, Any]:
        return deepcopy(self.resolved_model_config.get("training", {}))

    def build_model(self) -> GradientBoostingRegressor:
        params = deepcopy(self.resolved_model_config["parameters"])

        try:
            model = GradientBoostingRegressor(**params)
        except Exception as e:
            raise ModelConfigurationError(
                "Failed to construct GradientBoostingRegressor.",
                context={"error": str(e), "parameters": params},
            ) from e

        return model

    def fit(self, X, y, **kwargs) -> "GBRRunner":
        X_values, y_values = self.validate_X_y(X, y, allow_dataframe=True)

        failure_policy = self.resolved_model_config.get("failure_policy", {})
        if failure_policy.get("fail_on_empty_feature_matrix", True) and X_values.shape[1] == 0:
            raise ExperimentRuntimeError("Feature matrix has zero columns.")

        self.model = self.build_model()

        fit_kwargs: dict[str, Any] = {}
        training_cfg = self.resolved_model_config.get("training", {})
        use_sample_weights = bool(training_cfg.get("use_sample_weights", False))

        if use_sample_weights and "sample_weight" in kwargs:
            fit_kwargs["sample_weight"] = kwargs["sample_weight"]

        try:
            self.model.fit(X_values, y_values, **fit_kwargs)
        except Exception as e:
            raise ExperimentRuntimeError(
                "Failed to fit GradientBoostingRegressor.",
                context={"error": str(e)},
            ) from e

        self.is_fitted = True
        self._populate_artifacts_after_fit()
        return self

    def predict(self, X) -> np.ndarray:
        self.check_is_fitted()
        X_values = self.validate_X(X, allow_dataframe=True)

        try:
            preds = self.model.predict(X_values)
        except Exception as e:
            raise ExperimentRuntimeError(
                "Failed to generate predictions with GradientBoostingRegressor.",
                context={"error": str(e)},
            ) from e

        preds = np.asarray(preds, dtype=float).reshape(-1)

        if np.isnan(preds).any():
            raise ExperimentRuntimeError(
                "Predictions contain NaN values.",
                context={"nan_count": int(np.isnan(preds).sum())},
            )

        if np.isinf(preds).any():
            raise ExperimentRuntimeError(
                "Predictions contain infinite values.",
                context={"inf_count": int(np.isinf(preds).sum())},
            )

        return preds

    def _populate_artifacts_after_fit(self) -> None:
        if self.model is None:
            return

        artifact_cfg = self.resolved_model_config.get("artifacts", {})

        if artifact_cfg.get("save_feature_importance", True):
            importances = getattr(self.model, "feature_importances_", None)
            if importances is not None:
                importances = np.asarray(importances, dtype=float).reshape(-1)
                feature_names = self.get_feature_names()
                if feature_names and len(feature_names) == len(importances):
                    self.artifacts.feature_importance = {
                        feature_name: float(value)
                        for feature_name, value in zip(feature_names, importances)
                    }
                else:
                    self.artifacts.feature_importance = {
                        f"feature_{i}": float(value)
                        for i, value in enumerate(importances)
                    }

        self.artifacts.extra["resolved_model_config"] = deepcopy(self.resolved_model_config)
        self.artifacts.extra["n_features_in_"] = getattr(self.model, "n_features_in_", None)
        self.artifacts.extra["train_score_last"] = (
            float(self.model.train_score_[-1])
            if hasattr(self.model, "train_score_") and len(self.model.train_score_) > 0
            else None
        )
from dataclasses import dataclass


@dataclass(slots=True)
class GBRModelConfig:
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = 3
    random_state: int = 42
    loss: str = "squared_error"
    subsample: float = 1.0
    criterion: str = "friedman_mse"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    min_impurity_decrease: float = 0.0
    init: Any = None
    max_features: str | int | float | None = None
    alpha: float = 0.9
    verbose: int = 0
    max_leaf_nodes: int | None = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int | None = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0


def build_gbr_model(**model_params: Any) -> GBRRunner:
    """
    Factory expected by model_factory.py.

    It wraps flat model params into the experiment-style config
    that GBRRunner already knows how to resolve.
    """
    cfg = {
        "model_name": "gbr",
        "model_overrides": {
            "parameters": dict(model_params),
        },
    }
    return GBRRunner(config=cfg)


def get_gbr_importances(
    fitted_model: Any,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """
    Return (feature_name, importance) pairs for a fitted GBR model or runner.
    """
    model = fitted_model.model if hasattr(fitted_model, "model") else fitted_model

    if model is None:
        raise ValueError("GBR model is not fitted yet.")

    if not hasattr(model, "feature_importances_"):
        raise ValueError("GBR model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_, dtype=float).reshape(-1)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(importances)} importances vs "
            f"{len(feature_names)} feature names."
        )

    return list(zip(feature_names, importances.tolist()))