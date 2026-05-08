# src/models/ridge_model.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RidgeModelConfig:
    """
    Configuration for Ridge regression pipeline.

    Notes
    -----
    - Numeric features:
        median imputation -> scaling
    - Categorical features:
        most_frequent imputation -> one-hot encoding
    - If `use_cv=True`, RidgeCV is used to select alpha internally.
    - If `use_cv=False`, plain Ridge is used with the provided alpha.
    """
    alpha: float = 1.0
    alphas: Sequence[float] = field(
        default_factory=lambda: (
            1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0
        )
    )
    use_cv: bool = True
    fit_intercept: bool = True
    solver: str = "auto"
    max_iter: int = 10000
    tol: float = 1e-4
    random_state: int = 42


def _make_numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
                    sparse_output=False,  # dense output for simpler downstream handling
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
            "No features were provided to Ridge model. "
            "At least one numeric or categorical feature is required."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_estimator(cfg: RidgeModelConfig):
    if cfg.use_cv:
        return RidgeCV(
            alphas=np.asarray(cfg.alphas, dtype=float),
            fit_intercept=cfg.fit_intercept,
            scoring="neg_root_mean_squared_error",
        )

    return Ridge(
        alpha=cfg.alpha,
        fit_intercept=cfg.fit_intercept,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
    )


def build_ridge_model(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    cfg: RidgeModelConfig | None = None,
) -> Pipeline:
    """
    Build a full Ridge regression pipeline.

    Parameters
    ----------
    numeric_features:
        List of numeric feature column names.
    categorical_features:
        List of categorical feature column names.
    cfg:
        RidgeModelConfig instance. If None, defaults are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Full pipeline: preprocessing + ridge estimator
    """
    cfg = cfg or RidgeModelConfig()
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


def get_ridge_feature_names(model: Pipeline) -> list[str]:
    """
    Return transformed feature names after preprocessing.

    Useful for coefficient inspection after fitting.
    """
    if "preprocessor" not in model.named_steps:
        raise ValueError("Model has no 'preprocessor' step.")

    preprocessor = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_ridge_coefficients(model: Pipeline) -> list[tuple[str, float]]:
    """
    Return (feature_name, coefficient) pairs from a fitted Ridge model.
    """
    if "regressor" not in model.named_steps:
        raise ValueError("Model has no 'regressor' step.")

    regressor = model.named_steps["regressor"]
    if not hasattr(regressor, "coef_"):
        raise ValueError("Regressor is not fitted yet or has no coefficients.")

    feature_names = get_ridge_feature_names(model)
    coefs = regressor.coef_

    if len(feature_names) != len(coefs):
        raise ValueError(
            f"Feature/coefficient mismatch: {len(feature_names)} names vs {len(coefs)} coefficients."
        )

    pairs = list(zip(feature_names, coefs))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


def get_selected_alpha(model: Pipeline) -> float | None:
    """
    Return the selected alpha if RidgeCV was used.
    """
    regressor = model.named_steps.get("regressor")
    if regressor is None:
        raise ValueError("Model has no 'regressor' step.")

    return getattr(regressor, "alpha_", None)