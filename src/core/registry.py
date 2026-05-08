from __future__ import annotations

from typing import Any, Type

from src.core.exceptions import ConfigError
from src.models.base import BaseModelRunner
from src.models.gbr_model import GBRRunner


MODEL_REGISTRY: dict[str, Type[BaseModelRunner]] = {
    "gbr": GBRRunner,
    # "ridge": RidgeRunner,
    # "svr": SVRRunner,
    # "xgboost": XGBRunner,
    # "xlmr_text": XLMRTextRunner,
    # "hybrid": HybridRunner,
}

FEATURE_REGISTRY: dict[str, Any] = {}

METRIC_REGISTRY: dict[str, Any] = {}


def normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigError("'model_name' must be a non-empty string.")
    return model_name.strip().lower()


def get_model_runner_class(model_name: str) -> Type[BaseModelRunner]:
    normalized = normalize_model_name(model_name)

    if normalized not in MODEL_REGISTRY:
        raise ConfigError(
            f"Unsupported model_name '{model_name}'. "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[normalized]


def build_model_runner(cfg: dict[str, Any]) -> BaseModelRunner:
    if not isinstance(cfg, dict):
        raise ConfigError(
            "Experiment config must be a dictionary to build a model runner."
        )

    if "model_name" not in cfg:
        raise ConfigError("Missing required field 'model_name' in experiment config.")

    runner_cls = get_model_runner_class(cfg["model_name"])
    return runner_cls(cfg)