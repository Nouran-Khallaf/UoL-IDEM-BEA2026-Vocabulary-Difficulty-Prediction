from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


_VALID_SVR_KERNELS = {"linear", "poly", "rbf", "sigmoid", "precomputed"}


@dataclass(slots=True)
class SVRModelConfig:
    """
    Configuration for Support Vector Regression.

    Notes
    -----
    - SVR is sensitive to feature scale, so numeric scaling is essential.
    - For categorical features, one-hot expansion is followed by scaling so that
      the induced binary dimensions remain numerically compatible with the
      numeric branch.
    - The default kernel is RBF, which is the most important setting for your
      use case and aligns with common regression practice in quality-estimation
      style tasks.
    """
    kernel: str = "rbf"
    degree: int = 3
    gamma: str | float = "scale"
    C: float = 1.0
    epsilon: float = 0.1
    coef0: float = 0.0
    shrinking: bool = True
    tol: float = 1e-3
    cache_size: float = 500.0
    max_iter: int = -1


def _validate_svr_config(cfg: SVRModelConfig) -> None:
    kernel = cfg.kernel.strip().lower()
    if kernel not in _VALID_SVR_KERNELS:
        raise ValueError(
            f"Unsupported SVR kernel '{cfg.kernel}'. "
            f"Expected one of: {sorted(_VALID_SVR_KERNELS)}"
        )

    if cfg.C <= 0:
        raise ValueError("SVR parameter 'C' must be > 0.")
    if cfg.epsilon < 0:
        raise ValueError("SVR parameter 'epsilon' must be >= 0.")
    if cfg.degree < 0:
        raise ValueError("SVR parameter 'degree' must be >= 0.")
    if cfg.tol <= 0:
        raise ValueError("SVR parameter 'tol' must be > 0.")
    if cfg.cache_size <= 0:
        raise ValueError("SVR parameter 'cache_size' must be > 0.")

    if not (
        isinstance(cfg.gamma, (int, float))
        or cfg.gamma in {"scale", "auto"}
    ):
        raise ValueError(
            "SVR parameter 'gamma' must be a float or one of {'scale', 'auto'}."
        )


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
                    sparse_output=False,
                ),
            ),
            # Keep categorical branch on a comparable scale for SVR.
            ("scaler", StandardScaler()),
        ]
    )


def _make_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
) -> ColumnTransformer:
    numeric_features = list(numeric_features or [])
    categorical_features = list(categorical_features or [])

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(("num", _make_numeric_pipeline(), numeric_features))

    if categorical_features:
        transformers.append(("cat", _make_categorical_pipeline(), categorical_features))

    if not transformers:
        raise ValueError(
            "No features were provided to the SVR model. "
            "At least one numeric or categorical feature is required."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_estimator(cfg: SVRModelConfig) -> SVR:
    _validate_svr_config(cfg)

    return SVR(
        kernel=cfg.kernel,
        degree=cfg.degree,
        gamma=cfg.gamma,
        C=cfg.C,
        epsilon=cfg.epsilon,
        coef0=cfg.coef0,
        shrinking=cfg.shrinking,
        tol=cfg.tol,
        cache_size=cfg.cache_size,
        max_iter=cfg.max_iter,
    )


def build_svr_model(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    cfg: SVRModelConfig | None = None,
) -> Pipeline:
    """
    Build a full preprocessing + SVR regression pipeline.

    Parameters
    ----------
    numeric_features:
        Names of numeric input columns.
    categorical_features:
        Names of categorical input columns.
    cfg:
        Typed SVRModelConfig. If None, defaults are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with:
        - preprocessor
        - regressor
    """
    cfg = cfg or SVRModelConfig()
    preprocessor = _make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    estimator = _make_estimator(cfg)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )


def build_svr_pipeline(
    *,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Convenience wrapper used by the shared model factory.

    This mirrors the interface used by the other tabular models in the project,
    so the experiment runner can pass config-driven parameters directly.

    Parameters
    ----------
    numeric_features:
        Numeric feature column names.
    categorical_features:
        Categorical feature column names.
    model_params:
        Flat dictionary from YAML/config resolution.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit SVR pipeline.
    """
    params = dict(model_params or {})
    cfg = SVRModelConfig(
        kernel=params.get("kernel", "rbf"),
        degree=int(params.get("degree", 3)),
        gamma=params.get("gamma", "scale"),
        C=float(params.get("C", 1.0)),
        epsilon=float(params.get("epsilon", 0.1)),
        coef0=float(params.get("coef0", 0.0)),
        shrinking=bool(params.get("shrinking", True)),
        tol=float(params.get("tol", 1e-3)),
        cache_size=float(params.get("cache_size", 500.0)),
        max_iter=int(params.get("max_iter", -1)),
    )

    return build_svr_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )


def get_svr_feature_names(model: Pipeline) -> list[str]:
    """
    Return transformed feature names from the fitted preprocessor.

    This is useful for diagnostics and consistency with the existing
    feature-importance utilities, even though vanilla SVR itself does not expose
    coefficient-style importance for non-linear kernels.
    """
    if "preprocessor" not in model.named_steps:
        raise ValueError("Model has no 'preprocessor' step.")

    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("Preprocessor does not expose get_feature_names_out().")

    return list(preprocessor.get_feature_names_out())