from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.core.exceptions import ConfigError, DataValidationError


ALLOWED_CV_SCHEMES = {"kfold", "groupkfold", "stratifiedkfold"}


@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_size": int(len(self.train_idx)),
            "valid_size": int(len(self.valid_idx)),
        }


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a dictionary, got {type(value).__name__}.")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean, got {type(value).__name__}.")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer, got {value!r}.")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _normalize_cv_scheme(cv_cfg: dict[str, Any]) -> str:
    scheme = cv_cfg.get("scheme", cv_cfg.get("splitter"))
    if scheme is None:
        raise ConfigError("CV config must define either 'cv.scheme' or 'cv.splitter'.")

    scheme = _require_nonempty_string(scheme, "cv.scheme").lower()
    if scheme not in ALLOWED_CV_SCHEMES:
        raise ConfigError(
            f"Unsupported CV scheme '{scheme}'. Allowed: {sorted(ALLOWED_CV_SCHEMES)}"
        )
    return scheme


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            "Expected a pandas DataFrame for splitting.",
            context={"received_type": type(df).__name__},
        )
    if df.empty:
        raise DataValidationError("Cannot create CV splits from an empty DataFrame.")


def _validate_fold_count(n_samples: int, n_splits: int) -> None:
    if n_splits < 2:
        raise ConfigError(f"'cv.folds' must be at least 2, got {n_splits}.")
    if n_samples < n_splits:
        raise DataValidationError(
            "Number of CV folds cannot exceed number of samples.",
            context={"n_samples": n_samples, "n_splits": n_splits},
        )


def _extract_splitter_params(cv_cfg: dict[str, Any]) -> dict[str, Any]:
    n_splits = _require_positive_int(cv_cfg.get("folds"), "cv.folds")
    shuffle = _require_bool(cv_cfg.get("shuffle", True), "cv.shuffle")

    random_state = cv_cfg.get("random_state", None)
    if random_state is not None and not isinstance(random_state, int):
        raise ConfigError("'cv.random_state' must be an integer or null.")

    group_column = cv_cfg.get("group_column", None)
    if group_column is not None:
        group_column = _require_nonempty_string(group_column, "cv.group_column")

    stratify = cv_cfg.get("stratify", False)
    stratify = _require_bool(stratify, "cv.stratify")

    return {
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
        "group_column": group_column,
        "stratify": stratify,
    }


def _build_kfold_splitter(
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
) -> KFold:
    if not shuffle:
        random_state = None

    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def _build_groupkfold_splitter(
    *,
    n_splits: int,
) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


def _build_stratifiedkfold_splitter(
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int | None,
) -> StratifiedKFold:
    if not shuffle:
        random_state = None

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def _validate_group_column(df: pd.DataFrame, group_column: str) -> None:
    if group_column not in df.columns:
        raise DataValidationError(
            "Configured group_column is not present in the DataFrame.",
            context={"group_column": group_column, "available_columns": list(df.columns)},
        )

    if df[group_column].isna().any():
        raise DataValidationError(
            "group_column contains missing values, which is not allowed for GroupKFold.",
            context={
                "group_column": group_column,
                "missing_count": int(df[group_column].isna().sum()),
            },
        )


def _build_stratify_labels(
    df: pd.DataFrame,
    *,
    target_column: str,
) -> np.ndarray:
    """
    Build discrete labels for stratified splitting.
    Since this task is regression, StratifiedKFold should be used cautiously.
    We bin the target into quantile bins as an approximation.
    """
    if target_column not in df.columns:
        raise DataValidationError(
            "target_column not found for stratified splitting.",
            context={"target_column": target_column},
        )

    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.isna().any():
        raise DataValidationError(
            "Target column contains NaN values; cannot build stratification bins.",
            context={"target_column": target_column, "nan_count": int(y.isna().sum())},
        )

    # Conservative default: 5 quantile bins if possible
    n_bins = min(5, y.nunique())
    if n_bins < 2:
        raise DataValidationError(
            "Cannot stratify because the target does not have enough distinct values.",
            context={"target_column": target_column, "n_unique_target": int(y.nunique())},
        )

    try:
        labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except Exception as e:
        raise DataValidationError(
            "Failed to create quantile bins for stratified splitting.",
            context={"target_column": target_column, "error": str(e)},
        ) from e

    labels = np.asarray(labels)
    if np.isnan(labels).any():
        raise DataValidationError(
            "Stratification labels contain NaN values after qcut.",
            context={"target_column": target_column},
        )

    return labels.astype(int)


def build_folds(
    df: pd.DataFrame,
    cv_cfg: dict[str, Any],
    *,
    target_column: str | None = None,
) -> list[FoldIndices]:
    """
    Create fold indices according to CV config.

    Parameters
    ----------
    df:
        DataFrame to split.
    cv_cfg:
        CV configuration block.
    target_column:
        Required only for stratified regression splitting.

    Returns
    -------
    list[FoldIndices]
    """
    _validate_dataframe(df)
    cv_cfg = _require_dict(cv_cfg, "cv")

    scheme = _normalize_cv_scheme(cv_cfg)
    params = _extract_splitter_params(cv_cfg)

    n_samples = len(df)
    _validate_fold_count(n_samples, params["n_splits"])

    indices = np.arange(n_samples)
    folds: list[FoldIndices] = []

    if scheme == "kfold":
        splitter = _build_kfold_splitter(
            n_splits=params["n_splits"],
            shuffle=params["shuffle"],
            random_state=params["random_state"],
        )

        iterator = splitter.split(indices)

    elif scheme == "groupkfold":
        group_column = params["group_column"]
        if group_column is None:
            raise ConfigError("cv.group_column must be provided for groupkfold.")

        _validate_group_column(df, group_column)
        groups = df[group_column].to_numpy()

        splitter = _build_groupkfold_splitter(
            n_splits=params["n_splits"],
        )
        iterator = splitter.split(indices, groups=groups)

    elif scheme == "stratifiedkfold":
        if target_column is None:
            raise ConfigError(
                "target_column must be provided to build stratified regression folds."
            )

        labels = _build_stratify_labels(df, target_column=target_column)
        splitter = _build_stratifiedkfold_splitter(
            n_splits=params["n_splits"],
            shuffle=params["shuffle"],
            random_state=params["random_state"],
        )
        iterator = splitter.split(indices, labels)

    else:
        raise ConfigError(f"Unhandled CV scheme: {scheme}")

    for fold_id, (train_idx, valid_idx) in enumerate(iterator, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        valid_idx = np.asarray(valid_idx, dtype=int)

        if train_idx.size == 0 or valid_idx.size == 0:
            raise DataValidationError(
                "Generated an empty train or validation fold.",
                context={
                    "fold_id": fold_id,
                    "train_size": int(train_idx.size),
                    "valid_size": int(valid_idx.size),
                },
            )

        overlap = np.intersect1d(train_idx, valid_idx)
        if overlap.size > 0:
            raise DataValidationError(
                "Train and validation indices overlap within a fold.",
                context={"fold_id": fold_id, "overlap_count": int(overlap.size)},
            )

        folds.append(
            FoldIndices(
                fold_id=fold_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
            )
        )

    if len(folds) != params["n_splits"]:
        raise DataValidationError(
            "Number of generated folds does not match requested n_splits.",
            context={
                "requested_n_splits": params["n_splits"],
                "generated_n_folds": len(folds),
            },
        )

    return folds


def iter_folds(
    df: pd.DataFrame,
    cv_cfg: dict[str, Any],
    *,
    target_column: str | None = None,
) -> Iterator[FoldIndices]:
    """
    Iterator wrapper around build_folds().
    """
    yield from build_folds(df, cv_cfg, target_column=target_column)


def summarize_folds(
    folds: list[FoldIndices],
    *,
    df: pd.DataFrame | None = None,
    id_column: str | None = None,
) -> list[dict[str, Any]]:
    """
    Create fold diagnostics for logging/reporting.
    """
    summary: list[dict[str, Any]] = []

    for fold in folds:
        record: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "train_size": int(len(fold.train_idx)),
            "valid_size": int(len(fold.valid_idx)),
        }

        if df is not None:
            record["n_total"] = int(len(df))

            if id_column is not None and id_column in df.columns:
                train_ids = df.iloc[fold.train_idx][id_column]
                valid_ids = df.iloc[fold.valid_idx][id_column]
                overlap_ids = set(train_ids).intersection(set(valid_ids))
                record["id_overlap_count"] = int(len(overlap_ids))

        summary.append(record)

    return summary