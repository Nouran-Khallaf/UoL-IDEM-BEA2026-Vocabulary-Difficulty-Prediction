from __future__ import annotations

from typing import Any

from src.core.exceptions import ExperimentRuntimeError
from src.models.gbr_model import GBRModelConfig, build_gbr_model
from src.models.ridge_model import RidgeModelConfig, build_ridge_model
from src.models.svr_model import SVRModelConfig, build_svr_model
from src.models.xgb_model import XGBModelConfig, build_xgb_model


def _normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ExperimentRuntimeError("model_name must be a non-empty string.")

    name = model_name.strip().lower()

    if name == "ridge":
        return "ridge"

    if name in {"gbr", "gradientboostingregressor", "gradient_boosting"}:
        return "gbr"

    if name in {"xgb", "xgboost", "xgbregressor"}:
        return "xgboost"

    if name in {"svr", "supportvectorregression", "support_vector_regression"}:
        return "svr"

    raise ExperimentRuntimeError(
        f"Unsupported model_name '{model_name}'. "
        f"Supported: ['gbr', 'ridge', 'svr', 'xgb', 'xgboost']"
    )


def _build_ridge_from_params(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
):
    cfg = RidgeModelConfig(
        alpha=float(model_params.get("alpha", 1.0)),
        alphas=tuple(model_params.get(
            "alphas",
            (1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        )),
        use_cv=bool(model_params.get("use_cv", True)),
        fit_intercept=bool(model_params.get("fit_intercept", True)),
        solver=str(model_params.get("solver", "auto")),
        max_iter=int(model_params.get("max_iter", 10000)),
        tol=float(model_params.get("tol", 1e-4)),
        random_state=int(model_params.get("random_state", 42)),
    )
    return build_ridge_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )


def _build_gbr_from_params(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
):
    cfg = GBRModelConfig(
        n_estimators=int(model_params.get("n_estimators", 300)),
        learning_rate=float(model_params.get("learning_rate", 0.05)),
        max_depth=int(model_params.get("max_depth", 3)),
        min_samples_split=int(model_params.get("min_samples_split", 2)),
        min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
        subsample=float(model_params.get("subsample", 1.0)),
        max_features=model_params.get("max_features", None),
        loss=str(model_params.get("loss", "squared_error")),
        random_state=int(model_params.get("random_state", 42)),
    )
    return build_gbr_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )


def _build_xgb_from_params(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
):
    cfg = XGBModelConfig(
        n_estimators=int(model_params.get("n_estimators", 500)),
        max_depth=int(model_params.get("max_depth", 6)),
        learning_rate=float(model_params.get("learning_rate", 0.05)),
        subsample=float(model_params.get("subsample", 0.9)),
        colsample_bytree=float(model_params.get("colsample_bytree", 0.9)),
        reg_alpha=float(model_params.get("reg_alpha", 0.0)),
        reg_lambda=float(model_params.get("reg_lambda", 1.0)),
        min_child_weight=float(model_params.get("min_child_weight", 1.0)),
        gamma=float(model_params.get("gamma", 0.0)),
        objective=str(model_params.get("objective", "reg:squarederror")),
        eval_metric=str(model_params.get("eval_metric", "rmse")),
        n_jobs=int(model_params.get("n_jobs", -1)),
        random_state=int(model_params.get("random_state", 42)),
    )
    return build_xgb_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )


def _build_svr_from_params(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any],
):
    cfg = SVRModelConfig(
        kernel=str(model_params.get("kernel", "rbf")),
        degree=int(model_params.get("degree", 3)),
        gamma=model_params.get("gamma", "scale"),
        C=float(model_params.get("C", 1.0)),
        epsilon=float(model_params.get("epsilon", 0.1)),
        coef0=float(model_params.get("coef0", 0.0)),
        shrinking=bool(model_params.get("shrinking", True)),
        tol=float(model_params.get("tol", 1e-3)),
        cache_size=float(model_params.get("cache_size", 500.0)),
        max_iter=int(model_params.get("max_iter", -1)),
    )
    return build_svr_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )


def build_model(
    *,
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict[str, Any] | None = None,
):
    """
    Build a tabular regression pipeline from config-driven parameters.

    Parameters
    ----------
    model_name:
        Name of the downstream regressor.
    numeric_features:
        Numeric feature column names.
    categorical_features:
        Categorical feature column names.
    model_params:
        Flat parameter dictionary resolved from YAML/config.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit regression pipeline.
    """
    resolved_name = _normalize_model_name(model_name)
    model_params = dict(model_params or {})

    if resolved_name == "ridge":
        return _build_ridge_from_params(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )

    if resolved_name == "gbr":
        return _build_gbr_from_params(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )

    if resolved_name == "xgboost":
        return _build_xgb_from_params(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )

    if resolved_name == "svr":
        return _build_svr_from_params(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )

    raise ExperimentRuntimeError(
        f"Could not build model for normalized name '{resolved_name}'."
    )