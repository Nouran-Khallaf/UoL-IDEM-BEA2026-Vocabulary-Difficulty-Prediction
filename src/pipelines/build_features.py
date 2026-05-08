from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd
import yaml

from src.core.exceptions import ExperimentRuntimeError, FeatureValidationError
from src.data.load_raw import load_raw_dataset


@dataclass(slots=True)
class FeatureBuildSplitResult:
    split_name: str
    df: pd.DataFrame
    diagnostics: dict[str, Any]
    feature_columns: list[str]
    feature_groups_applied: list[str]


@dataclass(slots=True)
class FeatureBuildResult:
    resolved_config: dict[str, Any]
    output_dir: Path
    splits: dict[str, FeatureBuildSplitResult]


_FEATURE_GROUP_TO_MODULE = {
    "frequency": "src.features.frequency",
    "lexical": "src.features.lexical",
    "retrieval": "src.features.retrieval",
    "mlm": "src.features.mlm",
    "surprisal": "src.features.surprisal",
    "cognate": "src.features.cognate",
    "semantic": "src.features.semantic",
    "mlm_predicted_word": "src.features.mlm_predicted_word",
    "baseline_meta": "src.features.baseline_meta",
}


# -------------------------------------------------
# Generic helpers
# -------------------------------------------------
def _ensure_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentRuntimeError(f"'{name}' must be a dictionary.")
    return value



def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged



def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    processed_dir = paths.get("processed_data_dir") or "data/processed"
    experiment_name = cfg.get("experiment_name") or "feature_build"
    return (Path(processed_dir) / str(experiment_name)).resolve()


# -------------------------------------------------
# Feature-group resolution
# -------------------------------------------------
def _resolve_enabled_feature_groups(cfg: dict[str, Any]) -> list[str]:
    feature_groups = cfg.get("feature_groups")
    if feature_groups is None:
        raise ExperimentRuntimeError("Resolved config must define 'feature_groups'.")

    if isinstance(feature_groups, list):
        groups: list[str] = []
        seen: set[str] = set()
        for value in feature_groups:
            if not isinstance(value, str) or not value.strip():
                raise ExperimentRuntimeError("Each feature group must be a non-empty string.")
            name = value.strip()
            if name not in seen:
                groups.append(name)
                seen.add(name)
        if not groups:
            raise ExperimentRuntimeError("No enabled feature groups found in list-style feature_groups.")
        return groups

    if isinstance(feature_groups, dict):
        enabled = feature_groups.get("enabled", [])
        disabled = set(feature_groups.get("disabled", []))
        if not isinstance(enabled, list):
            raise ExperimentRuntimeError("feature_groups.enabled must be a list.")
        groups: list[str] = []
        seen: set[str] = set()
        for value in enabled:
            if not isinstance(value, str) or not value.strip():
                raise ExperimentRuntimeError("Each enabled feature group must be a non-empty string.")
            name = value.strip()
            if name in disabled:
                continue
            if name not in seen:
                groups.append(name)
                seen.add(name)
        if not groups:
            raise ExperimentRuntimeError("No enabled feature groups remain after applying disabled list.")
        return groups

    raise ExperimentRuntimeError("feature_groups must be either a list or a dictionary.")



def _candidate_feature_config_paths(cfg: dict[str, Any], group_name: str) -> list[Path]:
    meta = cfg.get("_meta") if isinstance(cfg.get("_meta"), dict) else {}
    candidates: list[Path] = []

    experiment_config_path = meta.get("source_config") or meta.get("experiment_config_path")
    experiment_config_dir = meta.get("experiment_config_dir")

    if isinstance(experiment_config_dir, str) and experiment_config_dir:
        exp_dir = Path(experiment_config_dir)
        candidates.extend(
            [
                exp_dir / "configs" / "features" / f"{group_name}.yaml",
                exp_dir / "configs" / "features" / f"{group_name}.yml",
                exp_dir / "features" / f"{group_name}.yaml",
                exp_dir / "features" / f"{group_name}.yml",
            ]
        )

    if isinstance(experiment_config_path, str) and experiment_config_path:
        src = Path(experiment_config_path)
        project_root = src.parent.parent.parent if len(src.parents) >= 3 else src.parent
        candidates.extend(
            [
                project_root / "configs" / "features" / f"{group_name}.yaml",
                project_root / "configs" / "features" / f"{group_name}.yml",
            ]
        )

    candidates.extend(
        [
            Path("configs") / "features" / f"{group_name}.yaml",
            Path("configs") / "features" / f"{group_name}.yml",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped



def _load_yaml_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to read YAML from {path}: {e}") from e
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ExperimentRuntimeError(f"Feature config at {path} must be a dictionary.")
    return loaded



def _load_feature_group_config(cfg: dict[str, Any], group_name: str) -> dict[str, Any]:
    for candidate in _candidate_feature_config_paths(cfg, group_name):
        loaded = _load_yaml_if_exists(candidate)
        if loaded is not None:
            return loaded
    return {}



def _get_feature_override(cfg: dict[str, Any], group_name: str) -> dict[str, Any]:
    feature_overrides = cfg.get("feature_overrides")
    if feature_overrides is None:
        return {}
    feature_overrides = _ensure_dict(feature_overrides, "feature_overrides")
    override = feature_overrides.get(group_name, {})
    if override is None:
        return {}
    if not isinstance(override, dict):
        raise ExperimentRuntimeError(
            f"feature_overrides.{group_name} must be a dictionary when provided."
        )
    return override



def _resolve_feature_group_config(cfg: dict[str, Any], group_name: str) -> dict[str, Any]:
    base_cfg = _load_feature_group_config(cfg, group_name)
    override_cfg = _get_feature_override(cfg, group_name)
    merged = _deep_merge_dicts(base_cfg, override_cfg)
    if "group_name" not in merged:
        merged["group_name"] = group_name
    if "enabled" not in merged:
        merged["enabled"] = True
    return merged


# -------------------------------------------------
# Feature execution
# -------------------------------------------------
def _default_feature_builder(
    df: pd.DataFrame,
    *,
    group_name: str,
    cfg: dict[str, Any],
    feature_group_cfg: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    expected = feature_group_cfg.get("columns_expected", [])
    if expected is not None:
        if not isinstance(expected, list):
            raise FeatureValidationError(
                f"Feature config for group '{group_name}' must define columns_expected as a list."
            )
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise FeatureValidationError(
                f"Feature group '{group_name}' is missing expected columns in split '{split_name}': {missing}"
            )
    return df.copy()



def _call_feature_module(
    df: pd.DataFrame,
    *,
    group_name: str,
    cfg: dict[str, Any],
    feature_group_cfg: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    module_path = _FEATURE_GROUP_TO_MODULE.get(group_name)
    if module_path is None:
        raise ExperimentRuntimeError(
            f"Unknown feature group '{group_name}'. No module mapping exists."
        )

    try:
        import importlib
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ExperimentRuntimeError(
            f"Failed to import module '{module_path}' for feature group '{group_name}': {e}"
        ) from e

    for fn_name in ("build_features", "build_feature_group", "build"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            try:
                return fn(
                    df=df,
                    cfg=cfg,
                    feature_group_cfg=feature_group_cfg,
                    split_name=split_name,
                )
            except TypeError:
                try:
                    return fn(df)
                except Exception as e:
                    raise ExperimentRuntimeError(
                        f"Feature group '{group_name}' failed in split '{split_name}': {e}"
                    ) from e
            except Exception as e:
                raise ExperimentRuntimeError(
                    f"Feature group '{group_name}' failed in split '{split_name}': {e}"
                ) from e

    return _default_feature_builder(
        df,
        group_name=group_name,
        cfg=cfg,
        feature_group_cfg=feature_group_cfg,
        split_name=split_name,
    )


# -------------------------------------------------
# Build orchestration
# -------------------------------------------------
def _resolve_feature_columns(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    target_column: str | None,
    id_column: str | None,
) -> list[str]:
    before_cols = set(df_before.columns)
    excluded = {c for c in (target_column, id_column) if c}

    new_columns = [col for col in df_after.columns if col not in before_cols and col not in excluded]
    if new_columns:
        return new_columns

    candidate_cols = [col for col in df_after.columns if col not in excluded]
    return candidate_cols



def _build_split_result(
    *,
    split_name: str,
    raw_df: pd.DataFrame,
    raw_diagnostics: dict[str, Any],
    cfg: dict[str, Any],
    enabled_groups: list[str],
) -> FeatureBuildSplitResult:
    current_df = raw_df.copy()
    applied_groups: list[str] = []
    per_group_meta: dict[str, Any] = {}

    schema_cfg = _ensure_dict(cfg.get("schema"), "schema")
    target_column = schema_cfg.get("target_column") if isinstance(schema_cfg.get("target_column"), str) else None
    id_column = schema_cfg.get("id_column") if isinstance(schema_cfg.get("id_column"), str) else None

    protected_columns = [
        "L1_source_word_raw",
        "L1_source_word",
        "L1_source_word_excluded_word",
        "L1_source_word_has_alternative",
        "L1_source_word_alternatives",
        "L1_context_excluded_word",
        "L1_context_has_excluded_word",
    ]
    protected_snapshot = {
        col: raw_df[col].copy()
        for col in protected_columns
        if col in raw_df.columns
    }

    for group_name in enabled_groups:
        feature_group_cfg = _resolve_feature_group_config(cfg, group_name)
        if not bool(feature_group_cfg.get("enabled", True)):
            continue

        before_df = current_df.copy()
        current_df = _call_feature_module(
            current_df,
            group_name=group_name,
            cfg=cfg,
            feature_group_cfg=feature_group_cfg,
            split_name=split_name,
        )
        if not isinstance(current_df, pd.DataFrame):
            raise ExperimentRuntimeError(
                f"Feature group '{group_name}' returned a non-DataFrame object for split '{split_name}'."
            )

        for col, original_series in protected_snapshot.items():
            if col in current_df.columns and not current_df[col].equals(original_series):
                raise FeatureValidationError(
                    f"Feature group '{group_name}' modified protected column '{col}' in split '{split_name}'."
                )

        group_feature_columns = _resolve_feature_columns(
            before_df,
            current_df,
            target_column=target_column,
            id_column=id_column,
        )
        per_group_meta[group_name] = {
            "group_config": feature_group_cfg,
            "group_feature_columns": group_feature_columns,
        }
        applied_groups.append(group_name)

    feature_columns = _resolve_feature_columns(
        raw_df,
        current_df,
        target_column=target_column,
        id_column=id_column,
    )

    diagnostics = dict(raw_diagnostics)
    diagnostics.update(
        {
            "n_rows": int(len(current_df)),
            "n_columns": int(current_df.shape[1]),
            "columns": list(current_df.columns),
            "n_rows_before_features": int(len(raw_df)),
            "n_columns_before_features": int(raw_df.shape[1]),
            "columns_before_features": list(raw_df.columns),
            "n_rows_after_features": int(len(current_df)),
            "n_columns_after_features": int(current_df.shape[1]),
            "columns_after_features": list(current_df.columns),
            "feature_groups_applied": applied_groups,
            "feature_group_details": per_group_meta,
            "n_feature_columns": len(feature_columns),
            "feature_columns": feature_columns,
        }
    )

    return FeatureBuildSplitResult(
        split_name=split_name,
        df=current_df,
        diagnostics=diagnostics,
        feature_columns=feature_columns,
        feature_groups_applied=applied_groups,
    )

def _save_split_result(output_dir: Path, split_result: FeatureBuildSplitResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = split_result.split_name

    split_result.df.to_csv(output_dir / f"{split_name}_features.csv", index=False)

    with (output_dir / f"{split_name}_feature_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(split_result.diagnostics, f, indent=2, ensure_ascii=False)



def _save_build_manifest(output_dir: Path, result: FeatureBuildResult) -> None:
    payload = {
        "experiment_name": result.resolved_config.get("experiment_name"),
        "feature_groups": _resolve_enabled_feature_groups(result.resolved_config),
        "splits": {
            split_name: {
                "n_rows": int(split_result.df.shape[0]),
                "n_columns": int(split_result.df.shape[1]),
                "n_feature_columns": int(len(split_result.feature_columns)),
                "feature_groups_applied": split_result.feature_groups_applied,
            }
            for split_name, split_result in result.splits.items()
        },
    }

    with (output_dir / "feature_build_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def build_features_pipeline(
    resolved_config: dict[str, Any],
    *,
    splits_to_build: list[str] | None = None,
    save_outputs: bool = True,
    output_dir: str | Path | None = None,
) -> FeatureBuildResult:
    """
    Execute the modular feature-building pipeline over configured data splits.

    Key behavior
    ------------
    - loads splits via src.data.load_raw.load_raw_dataset
    - resolves enabled feature groups from config
    - loads base feature YAML from configs/features/<group>.yaml when available
    - merges per-experiment feature_overrides.<group> on top of the base feature YAML
    - calls each feature module with the merged feature_group_cfg
    - writes per-split feature tables + diagnostics + manifest
    """
    if not isinstance(resolved_config, dict):
        raise ExperimentRuntimeError("resolved_config must be a dictionary.")

    enabled_groups = _resolve_enabled_feature_groups(resolved_config)
    loaded = load_raw_dataset(resolved_config)

    requested_splits = splits_to_build or list(loaded.keys())
    requested_splits = [str(split).strip().lower() for split in requested_splits]

    invalid = [split for split in requested_splits if split not in loaded]
    if invalid:
        raise ExperimentRuntimeError(
            f"Requested splits are not available from load_raw_dataset: {invalid}. "
            f"Available: {sorted(loaded.keys())}"
        )

    resolved_output_dir = Path(output_dir).resolve() if output_dir is not None else _resolve_output_dir(resolved_config)

    split_results: dict[str, FeatureBuildSplitResult] = {}
    for split_name in requested_splits:
        payload = loaded[split_name]
        split_results[split_name] = _build_split_result(
            split_name=split_name,
            raw_df=payload["df"],
            raw_diagnostics=payload["diagnostics"],
            cfg=resolved_config,
            enabled_groups=enabled_groups,
        )

    result = FeatureBuildResult(
        resolved_config=resolved_config,
        output_dir=resolved_output_dir,
        splits=split_results,
    )

    if save_outputs:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        for split_result in result.splits.values():
            _save_split_result(resolved_output_dir, split_result)
        _save_build_manifest(resolved_output_dir, result)

    return result
