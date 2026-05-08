from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.core.exceptions import MetricComputationError


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _to_1d_numpy(x, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise MetricComputationError(
            f"Failed to convert '{name}' to a numeric numpy array.",
            context={"name": name, "error": str(e)},
        ) from e

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)

    return arr


def _validate_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = _to_1d_numpy(y_true, name="y_true")
    y_pred_arr = _to_1d_numpy(y_pred, name="y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise MetricComputationError(
            "y_true and y_pred must have the same number of elements.",
            context={
                "len_y_true": int(y_true_arr.shape[0]),
                "len_y_pred": int(y_pred_arr.shape[0]),
            },
        )

    if y_true_arr.shape[0] == 0:
        raise MetricComputationError("Metric inputs must not be empty.")

    if np.isnan(y_true_arr).any():
        raise MetricComputationError(
            "y_true contains NaN values.",
            context={"nan_count": int(np.isnan(y_true_arr).sum())},
        )

    if np.isnan(y_pred_arr).any():
        raise MetricComputationError(
            "y_pred contains NaN values.",
            context={"nan_count": int(np.isnan(y_pred_arr).sum())},
        )

    if np.isinf(y_true_arr).any():
        raise MetricComputationError(
            "y_true contains infinite values.",
            context={"inf_count": int(np.isinf(y_true_arr).sum())},
        )

    if np.isinf(y_pred_arr).any():
        raise MetricComputationError(
            "y_pred contains infinite values.",
            context={"inf_count": int(np.isinf(y_pred_arr).sum())},
        )

    return y_true_arr, y_pred_arr


def _is_constant(arr: np.ndarray) -> bool:
    return bool(np.allclose(arr, arr[0]))


def rmse(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    try:
        return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    except Exception as e:
        raise MetricComputationError(
            "Failed to compute RMSE.",
            context={"error": str(e)},
        ) from e


def mae(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    try:
        return float(mean_absolute_error(y_true_arr, y_pred_arr))
    except Exception as e:
        raise MetricComputationError(
            "Failed to compute MAE.",
            context={"error": str(e)},
        ) from e


def pearson(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    if _is_constant(y_true_arr):
        raise MetricComputationError(
            "Pearson correlation is undefined because y_true is constant."
        )

    if _is_constant(y_pred_arr):
        raise MetricComputationError(
            "Pearson correlation is undefined because y_pred is constant."
        )

    try:
        value, _ = pearsonr(y_true_arr, y_pred_arr)
        if np.isnan(value):
            raise MetricComputationError("Pearson correlation returned NaN.")
        return float(value)
    except MetricComputationError:
        raise
    except Exception as e:
        raise MetricComputationError(
            "Failed to compute Pearson correlation.",
            context={"error": str(e)},
        ) from e


def spearman(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    if _is_constant(y_true_arr):
        raise MetricComputationError(
            "Spearman correlation is undefined because y_true is constant."
        )

    if _is_constant(y_pred_arr):
        raise MetricComputationError(
            "Spearman correlation is undefined because y_pred is constant."
        )

    try:
        value, _ = spearmanr(y_true_arr, y_pred_arr)
        if np.isnan(value):
            raise MetricComputationError("Spearman correlation returned NaN.")
        return float(value)
    except MetricComputationError:
        raise
    except Exception as e:
        raise MetricComputationError(
            "Failed to compute Spearman correlation.",
            context={"error": str(e)},
        ) from e


def kendall_tau(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    if _is_constant(y_true_arr):
        raise MetricComputationError(
            "Kendall tau is undefined because y_true is constant."
        )

    if _is_constant(y_pred_arr):
        raise MetricComputationError(
            "Kendall tau is undefined because y_pred is constant."
        )

    try:
        value, _ = kendalltau(y_true_arr, y_pred_arr)
        if np.isnan(value):
            raise MetricComputationError("Kendall tau returned NaN.")
        return float(value)
    except MetricComputationError:
        raise
    except Exception as e:
        raise MetricComputationError(
            "Failed to compute Kendall tau.",
            context={"error": str(e)},
        ) from e


def normalize_metric_name(metric_name: str) -> str:
    metric_name = str(metric_name).strip().lower()
    if metric_name == "kendall":
        return "kendall_tau"
    return metric_name


METRIC_REGISTRY: dict[str, MetricFn] = {
    "rmse": rmse,
    "mae": mae,
    "pearson": pearson,
    "spearman": spearman,
    "kendall_tau": kendall_tau,
}


def get_metric_fn(metric_name: str) -> MetricFn:
    normalized = normalize_metric_name(metric_name)
    if normalized not in METRIC_REGISTRY:
        raise MetricComputationError(
            f"Unsupported metric '{metric_name}'.",
            context={"normalized_name": normalized, "allowed": sorted(METRIC_REGISTRY)},
        )
    return METRIC_REGISTRY[normalized]


def compute_metric(metric_name: str, y_true, y_pred) -> float:
    metric_fn = get_metric_fn(metric_name)
    return metric_fn(y_true, y_pred)


def compute_metrics(metric_names: list[str], y_true, y_pred) -> dict[str, float]:
    results: dict[str, float] = {}
    for metric_name in metric_names:
        normalized = normalize_metric_name(metric_name)
        results[normalized] = compute_metric(normalized, y_true, y_pred)
    return results