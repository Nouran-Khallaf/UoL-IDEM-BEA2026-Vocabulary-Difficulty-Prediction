# src/models/xgb_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


@dataclass
class XGBModelConfig:
    """
    Configuration for XGBoost regression pipeline.

    Notes
    -----
    - Numeric features:
        median imputation
    - Categorical features:
        most_frequent imputation -> one-hot encoding
    - Scaling is not needed for tree-based models.
    """
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    objective: str = "reg:squarederror"
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    max_bin: int = 256
    verbosity: int = 0


def _make_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


def _make_categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def _make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ColumnTransformer:
    numeric_features = list(numeric_features or [])
    categorical_features = list(categorical_features or [])

    transformers = []

    if numeric_features:
        transformers.append(("num", _make_numeric_pipeline(), numeric_features))

    if categorical_features:
        transformers.append(("cat", _make_categorical_pipeline(), categorical_features))

    if not transformers:
        raise ValueError(
            "No features were provided to XGB model. "
            "At least one numeric or categorical feature is required."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_estimator(cfg: XGBModelConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        gamma=cfg.gamma,
        objective=cfg.objective,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        max_bin=cfg.max_bin,
        verbosity=cfg.verbosity,
    )


def build_xgb_model(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    cfg: XGBModelConfig | None = None,
) -> Pipeline:
    """
    Build a full XGBoost regression pipeline.

    Parameters
    ----------
    numeric_features:
        List of numeric feature column names.
    categorical_features:
        List of categorical feature column names.
    cfg:
        XGBModelConfig instance. If None, defaults are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Full pipeline: preprocessing + XGBRegressor estimator
    """
    cfg = cfg or XGBModelConfig()
    categorical_features = list(categorical_features or [])

    preprocessor = _make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    estimator = _make_estimator(cfg)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )
    return model


def get_xgb_feature_names(model: Pipeline) -> list[str]:
    """
    Return transformed feature names after preprocessing.
    """
    if "preprocessor" not in model.named_steps:
        raise ValueError("Model has no 'preprocessor' step.")

    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_xgb_importances(model: Pipeline) -> list[tuple[str, float]]:
    """
    Return (feature_name, importance) pairs from a fitted XGB model.
    """
    if "regressor" not in model.named_steps:
        raise ValueError("Model has no 'regressor' step.")

    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        raise ValueError("Regressor is not fitted yet or has no feature importances.")

    feature_names = get_xgb_feature_names(model)
    importances = regressor.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature/importance mismatch: {len(feature_names)} names vs {len(importances)} importances."
        )

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs