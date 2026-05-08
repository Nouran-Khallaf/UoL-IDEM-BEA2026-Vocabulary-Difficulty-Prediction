from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.exceptions import ConfigError


ALLOWED_EXPERIMENT_TYPES = {
    "hybrid",
    "tabular",
    "text",
    "neural_fusion",
    "fusion_neural",
    "transformer_tabular_fusion",
    "text_prompt_regression"
}

ALLOWED_CV_SCHEMES = {
    "kfold",
    "groupkfold",
    "stratifiedkfold",
}

ALLOWED_METRICS = {
    "rmse",
    "mae",
    "pearson",
    "spearman",
    "kendall_tau",
    "kendall",  # alias, normalized later
}

ALLOWED_LOG_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}

ALLOWED_MODEL_NAMES = {
    "ridge",
    "gbr",
    "svr",
    "xgboost",
    "xgb",
    "xlmr_text",
    "hybrid",
    "neural_fusion",
    "text_prompt_regression",
}



def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a dictionary, got {type(value).__name__}.")
    return value



def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{name}' must be a list, got {type(value).__name__}.")
    return value



def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string.")
    return value.strip()



def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean, got {type(value).__name__}.")
    return value



def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer, got {value!r}.")
    return value



def _require_nonnegative_int_or_null(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{name}' must be null or a non-negative integer, got {value!r}.")
    return value



def normalize_metric_name(metric_name: str) -> str:
    metric_name = _require_nonempty_string(metric_name, "metric_name").lower()
    if metric_name == "kendall":
        return "kendall_tau"
    return metric_name



def normalize_model_name(model_name: str) -> str:
    model_name = _require_nonempty_string(model_name, "model_name").lower()
    if model_name == "xgb":
        return "xgboost"
    return model_name



def validate_metric_names(metric_names: list[str], *, field_name: str) -> list[str]:
    normalized: list[str] = []

    for metric in metric_names:
        metric = normalize_metric_name(metric)
        if metric not in ALLOWED_METRICS:
            raise ConfigError(
                f"Unsupported metric '{metric}' in '{field_name}'. "
                f"Allowed metrics: {sorted(ALLOWED_METRICS)}"
            )
        normalized.append(metric)

    deduped: list[str] = []
    seen: set[str] = set()
    for metric in normalized:
        if metric not in seen:
            deduped.append(metric)
            seen.add(metric)

    return deduped



def validate_paths_block(cfg: dict[str, Any]) -> None:
    paths = _require_dict(cfg.get("paths"), "paths")

    required_path_keys = [
        "raw_data_dir",
        "interim_data_dir",
        "processed_data_dir",
        "external_data_dir",
        "manifests_dir",
        "runs_dir",
    ]
    for key in required_path_keys:
        _require_nonempty_string(paths.get(key), f"paths.{key}")

    logs_dir = paths.get("logs_dir")
    if logs_dir is not None:
        _require_nonempty_string(logs_dir, "paths.logs_dir")



def validate_schema_block(cfg: dict[str, Any]) -> None:
    schema = _require_dict(cfg.get("schema"), "schema")

    _require_nonempty_string(schema.get("id_column"), "schema.id_column")
    _require_nonempty_string(schema.get("target_column"), "schema.target_column")
    _require_nonempty_string(schema.get("l1_column"), "schema.l1_column")



def validate_experiment_identity(cfg: dict[str, Any]) -> None:
    if "experiment_name" in cfg:
        _require_nonempty_string(cfg["experiment_name"], "experiment_name")

    if "experiment_type" in cfg:
        exp_type = _require_nonempty_string(cfg["experiment_type"], "experiment_type")
        if exp_type not in ALLOWED_EXPERIMENT_TYPES:
            raise ConfigError(
                f"Unsupported experiment_type '{exp_type}'. "
                f"Allowed: {sorted(ALLOWED_EXPERIMENT_TYPES)}"
            )



def validate_runtime_block(cfg: dict[str, Any]) -> None:
    runtime = cfg.get("runtime")
    if runtime is None:
        return

    runtime = _require_dict(runtime, "runtime")

    if "num_workers" in runtime:
        _require_positive_int(runtime["num_workers"], "runtime.num_workers")

    if "debug" in runtime:
        _require_bool(runtime["debug"], "runtime.debug")

    if "dry_run" in runtime:
        _require_bool(runtime["dry_run"], "runtime.dry_run")

    if "debug_subset_n" in runtime:
        _require_nonnegative_int_or_null(runtime["debug_subset_n"], "runtime.debug_subset_n")

    if "device" in runtime and runtime["device"] is not None:
        _require_nonempty_string(runtime["device"], "runtime.device")

    if "precision" in runtime and runtime["precision"] is not None:
        _require_nonempty_string(runtime["precision"], "runtime.precision")



def validate_logging_block(cfg: dict[str, Any]) -> None:
    logging_cfg = cfg.get("logging")
    if logging_cfg is None:
        return

    logging_cfg = _require_dict(logging_cfg, "logging")

    if "level" in logging_cfg:
        level = _require_nonempty_string(logging_cfg["level"], "logging.level").upper()
        if level not in ALLOWED_LOG_LEVELS:
            raise ConfigError(
                f"Unsupported logging.level '{level}'. "
                f"Allowed: {sorted(ALLOWED_LOG_LEVELS)}"
            )

    if "log_to_file" in logging_cfg:
        _require_bool(logging_cfg["log_to_file"], "logging.log_to_file")

    if "log_filename" in logging_cfg and logging_cfg["log_filename"] is not None:
        _require_nonempty_string(logging_cfg["log_filename"], "logging.log_filename")



def validate_feature_groups_block(cfg: dict[str, Any]) -> None:
    feature_groups = cfg.get("feature_groups")
    if feature_groups is None:
        return

    if isinstance(feature_groups, list):
        if len(feature_groups) == 0:
            raise ConfigError("'feature_groups' list cannot be empty.")
        for idx, value in enumerate(feature_groups):
            _require_nonempty_string(value, f"feature_groups[{idx}]")
        return

    feature_groups = _require_dict(feature_groups, "feature_groups")

    enabled = feature_groups.get("enabled", [])
    disabled = feature_groups.get("disabled", [])

    enabled = _require_list(enabled, "feature_groups.enabled")
    disabled = _require_list(disabled, "feature_groups.disabled")

    if len(enabled) == 0:
        raise ConfigError("'feature_groups.enabled' cannot be empty for an experiment.")

    enabled_set = set()
    for idx, value in enumerate(enabled):
        name = _require_nonempty_string(value, f"feature_groups.enabled[{idx}]")
        if name in enabled_set:
            raise ConfigError(f"Duplicate feature group in enabled list: '{name}'")
        enabled_set.add(name)

    for idx, value in enumerate(disabled):
        name = _require_nonempty_string(value, f"feature_groups.disabled[{idx}]")
        if name in enabled_set:
            raise ConfigError(
                f"Feature group '{name}' cannot appear in both enabled and disabled lists."
            )



def validate_cv_block(cfg: dict[str, Any]) -> None:
    cv = cfg.get("cv")
    if cv is None:
        return

    cv = _require_dict(cv, "cv")

    if "enabled" in cv:
        enabled = _require_bool(cv["enabled"], "cv.enabled")
    else:
        enabled = True

    if not enabled:
        return

    if "scheme" in cv:
        scheme = _require_nonempty_string(cv["scheme"], "cv.scheme").lower()
    elif "splitter" in cv:
        scheme = _require_nonempty_string(cv["splitter"], "cv.splitter").lower()
    else:
        raise ConfigError("CV config must define either 'cv.scheme' or 'cv.splitter'.")

    if scheme not in ALLOWED_CV_SCHEMES:
        raise ConfigError(
            f"Unsupported CV scheme '{scheme}'. Allowed: {sorted(ALLOWED_CV_SCHEMES)}"
        )

    if "folds" in cv:
        folds = _require_positive_int(cv["folds"], "cv.folds")
        if folds < 2:
            raise ConfigError("'cv.folds' must be at least 2.")
    else:
        raise ConfigError("Missing required field: 'cv.folds'")

    if "shuffle" in cv:
        _require_bool(cv["shuffle"], "cv.shuffle")

    if "random_state" in cv and cv["random_state"] is not None:
        if not isinstance(cv["random_state"], int):
            raise ConfigError("'cv.random_state' must be an integer or null.")

    if "stratify" in cv:
        _require_bool(cv["stratify"], "cv.stratify")

    if "group_column" in cv and cv["group_column"] is not None:
        _require_nonempty_string(cv["group_column"], "cv.group_column")



def validate_evaluation_block(cfg: dict[str, Any]) -> None:
    evaluation = cfg.get("evaluation")
    if evaluation is None:
        return

    evaluation = _require_dict(evaluation, "evaluation")

    if "metrics" in evaluation:
        metrics = _require_list(evaluation["metrics"], "evaluation.metrics")
        evaluation["metrics"] = validate_metric_names(metrics, field_name="evaluation.metrics")

    if "primary_metric" in evaluation:
        primary_metric = normalize_metric_name(evaluation["primary_metric"])
        if primary_metric not in ALLOWED_METRICS:
            raise ConfigError(
                f"Unsupported primary_metric '{primary_metric}'. "
                f"Allowed: {sorted(ALLOWED_METRICS)}"
            )
        evaluation["primary_metric"] = primary_metric

    if "secondary_metrics" in evaluation:
        secondary = _require_list(evaluation["secondary_metrics"], "evaluation.secondary_metrics")
        evaluation["secondary_metrics"] = validate_metric_names(
            secondary,
            field_name="evaluation.secondary_metrics",
        )

    if "compute_global_oof_metrics" in evaluation:
        _require_bool(
            evaluation["compute_global_oof_metrics"],
            "evaluation.compute_global_oof_metrics",
        )

    if "compute_fold_metrics" in evaluation:
        _require_bool(evaluation["compute_fold_metrics"], "evaluation.compute_fold_metrics")



def validate_outputs_block(cfg: dict[str, Any]) -> None:
    outputs = cfg.get("outputs")
    if outputs is None:
        return

    outputs = _require_dict(outputs, "outputs")

    for key, value in outputs.items():
        if key == "output_subdir":
            _require_nonempty_string(value, "outputs.output_subdir")
        elif key.startswith("save_"):
            _require_bool(value, f"outputs.{key}")



def validate_data_usage_block(cfg: dict[str, Any]) -> None:
    data_usage = cfg.get("data_usage")
    if data_usage is None:
        return

    data_usage = _require_dict(data_usage, "data_usage")

    allowed_run_modes = {
        "cross_validation_on_train",
        "train_and_eval_dev",
        "train_dev_for_test",
    }

    if "run_mode" in data_usage:
        run_mode = _require_nonempty_string(data_usage["run_mode"], "data_usage.run_mode")
        if run_mode not in allowed_run_modes:
            raise ConfigError(
                f"Unsupported data_usage.run_mode '{run_mode}'. "
                f"Allowed: {sorted(allowed_run_modes)}"
            )

    for split_name in ("train_split", "validation_split", "test_split"):
        if split_name in data_usage and data_usage[split_name] is not None:
            _require_nonempty_string(data_usage[split_name], f"data_usage.{split_name}")

    for bool_name in ("evaluate_dev_separately", "merge_train_dev_for_final_fit"):
        if bool_name in data_usage:
            _require_bool(data_usage[bool_name], f"data_usage.{bool_name}")



def validate_selection_block(cfg: dict[str, Any]) -> None:
    selection = cfg.get("selection")
    if selection is None:
        return

    selection = _require_dict(selection, "selection")

    if "target_column" in selection:
        _require_nonempty_string(selection["target_column"], "selection.target_column")

    if "id_column" in selection:
        _require_nonempty_string(selection["id_column"], "selection.id_column")

    if "keep_columns_always" in selection:
        keep_cols = _require_list(selection["keep_columns_always"], "selection.keep_columns_always")
        for idx, col in enumerate(keep_cols):
            _require_nonempty_string(col, f"selection.keep_columns_always[{idx}]")



def validate_model_block(cfg: dict[str, Any]) -> None:
    if "model_name" not in cfg:
        raise ConfigError("Missing required field: 'model_name'.")

    model_name = normalize_model_name(cfg["model_name"])
    if model_name not in ALLOWED_MODEL_NAMES:
        raise ConfigError(
            f"Unsupported model_name '{model_name}'. "
            f"Allowed: {sorted(ALLOWED_MODEL_NAMES)}"
        )

    cfg["model_name"] = model_name

    if "model_overrides" in cfg:
        model_overrides = _require_dict(cfg["model_overrides"], "model_overrides")

        for key in model_overrides.keys():
            if not isinstance(key, str) or not key.strip():
                raise ConfigError("All keys in 'model_overrides' must be non-empty strings.")



def validate_data_config_block(cfg: dict[str, Any]) -> None:
    if "files" in cfg:
        files = _require_dict(cfg["files"], "files")
        for split_name in ("train", "dev", "test"):
            if split_name in files and files[split_name] is not None:
                _require_nonempty_string(files[split_name], f"files.{split_name}")

    if "required_columns" in cfg:
        required_columns = _require_dict(cfg["required_columns"], "required_columns")
        for split_name, cols in required_columns.items():
            cols = _require_list(cols, f"required_columns.{split_name}")
            for idx, col in enumerate(cols):
                _require_nonempty_string(col, f"required_columns.{split_name}[{idx}]")

    if "validation" in cfg:
        validation = _require_dict(cfg["validation"], "validation")

        if "allowed_l1_values" in validation:
            vals = _require_list(validation["allowed_l1_values"], "validation.allowed_l1_values")
            for idx, item in enumerate(vals):
                _require_nonempty_string(item, f"validation.allowed_l1_values[{idx}]")

        for key, value in validation.items():
            if key == "allowed_l1_values":
                continue

            if key.startswith("forbid_") and isinstance(value, list):
                for idx, item in enumerate(value):
                    _require_nonempty_string(item, f"validation.{key}[{idx}]")
            elif (
                key.startswith("require_")
                or key.startswith("enforce_")
                or key.startswith("allow_")
                or key.startswith("forbid_")
            ):
                _require_bool(value, f"validation.{key}")



def validate_global_defaults(cfg: dict[str, Any]) -> None:
    defaults = cfg.get("defaults")
    if defaults is None:
        return

    defaults = _require_dict(defaults, "defaults")

    if "language" in defaults and defaults["language"] is not None:
        _require_nonempty_string(defaults["language"], "defaults.language")

    if "cv_folds" in defaults:
        _require_positive_int(defaults["cv_folds"], "defaults.cv_folds")

    if "cv_splitter" in defaults:
        splitter = _require_nonempty_string(defaults["cv_splitter"], "defaults.cv_splitter").lower()
        if splitter not in ALLOWED_CV_SCHEMES:
            raise ConfigError(
                f"Unsupported defaults.cv_splitter '{splitter}'. "
                f"Allowed: {sorted(ALLOWED_CV_SCHEMES)}"
            )

    if "cv_shuffle" in defaults:
        _require_bool(defaults["cv_shuffle"], "defaults.cv_shuffle")

    if "metrics" in defaults:
        metrics = _require_list(defaults["metrics"], "defaults.metrics")
        defaults["metrics"] = validate_metric_names(metrics, field_name="defaults.metrics")



def validate_resolved_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a fully resolved config dictionary.
    Returns the same config object, possibly with normalized names.
    """
    _require_dict(cfg, "resolved_config")

    validate_experiment_identity(cfg)
    validate_paths_block(cfg)
    validate_schema_block(cfg)
    validate_global_defaults(cfg)
    validate_runtime_block(cfg)
    validate_logging_block(cfg)
    validate_feature_groups_block(cfg)
    validate_cv_block(cfg)
    validate_evaluation_block(cfg)
    validate_outputs_block(cfg)
    validate_data_usage_block(cfg)
    validate_selection_block(cfg)
    validate_model_block(cfg)
    validate_data_config_block(cfg)

    return cfg



def validate_config_file_exists(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")
    return path.resolve()
