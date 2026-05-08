from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigError
from src.core.validation import validate_resolved_config


ALLOWED_CONFIG_EXTENSIONS = {".yaml", ".yml"}



def _ensure_dict(obj: Any, context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"{context} must be a dictionary, got {type(obj).__name__}.")
    return obj



def _read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    if path.suffix.lower() not in ALLOWED_CONFIG_EXTENSIONS:
        raise ConfigError(
            f"Unsupported config extension for {path}. "
            f"Expected one of: {sorted(ALLOWED_CONFIG_EXTENSIONS)}"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ConfigError(f"Failed to read config file {path}: {e}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e

    if data is None:
        data = {}

    return _ensure_dict(data, f"Root object in config {path}")



def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Rules:
    - scalar values in override replace base
    - lists in override replace base  (no list concatenation)
    - nested dicts merge recursively
    """
    merged = deepcopy(base)

    for key, override_value in override.items():
        if key not in merged:
            merged[key] = deepcopy(override_value)
            continue

        base_value = merged[key]

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge_dicts(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)

    return merged



def _normalize_inherits_block(
    raw_inherits: Any,
    *,
    config_path: Path,
) -> dict[str, str]:
    """
    Expected format:
        inherits:
          global: configs/global.yaml
          data: configs/data/de.yaml
          model: configs/models/ridge.yaml

    Returns a normalized dict[str, str].
    """
    if raw_inherits is None:
        return {}

    if not isinstance(raw_inherits, dict):
        raise ConfigError(
            f"'inherits' in {config_path} must be a mapping of name -> path."
        )

    normalized: dict[str, str] = {}
    for name, value in raw_inherits.items():
        if not isinstance(name, str):
            raise ConfigError(
                f"Invalid inherits key in {config_path}: expected string key, "
                f"got {type(name).__name__}."
            )
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"Invalid inherits value for key '{name}' in {config_path}: "
                f"expected non-empty string path."
            )
        normalized[name] = value.strip()

    return normalized



def _resolve_child_path(parent_config_path: Path, child_ref: str) -> Path:
    """
    Resolve inherited config path.

    Priority:
    1. absolute path -> use as-is
    2. relative path -> resolve from current working directory if it exists
    3. otherwise resolve relative to parent config file directory
    """
    ref_path = Path(child_ref)

    if ref_path.is_absolute():
        return ref_path

    cwd_candidate = Path.cwd() / ref_path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    return (parent_config_path.parent / ref_path).resolve()



def _resolve_inheritance_recursive(
    config_path: Path,
    *,
    visited: set[Path],
    stack: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    """
    Load config recursively with inheritance resolution.

    Returns:
        merged_config, lineage

    lineage records the resolved config files in merge order.
    """
    config_path = config_path.resolve()

    if config_path in stack:
        cycle = " -> ".join(str(p) for p in stack + [config_path])
        raise ConfigError(f"Cyclic config inheritance detected: {cycle}")

    current = _read_yaml_file(config_path)
    raw_inherits = current.pop("inherits", None)
    inherits_map = _normalize_inherits_block(raw_inherits, config_path=config_path)

    stack.append(config_path)

    merged: dict[str, Any] = {}
    lineage: list[str] = []

    for _, child_ref in inherits_map.items():
        child_path = _resolve_child_path(config_path, child_ref)
        child_cfg, child_lineage = _resolve_inheritance_recursive(
            child_path,
            visited=visited,
            stack=stack,
        )
        merged = _deep_merge_dicts(merged, child_cfg)
        lineage.extend(child_lineage)

    merged = _deep_merge_dicts(merged, current)
    lineage.append(str(config_path))

    stack.pop()
    visited.add(config_path)

    return merged, lineage



def _inject_metadata(
    cfg: dict[str, Any],
    *,
    source_path: Path,
    lineage: list[str],
) -> dict[str, Any]:
    enriched = deepcopy(cfg)
    enriched.setdefault("_meta", {})
    enriched["_meta"]["source_config"] = str(source_path.resolve())
    enriched["_meta"]["resolved_from"] = lineage
    return enriched



def normalize_resolved_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize scaffold-style keys into the richer resolved structure expected by
    the advanced pipeline.

    Examples
    --------
    - train_file/dev_file/test_file -> files.train/files.dev/files.test
    - columns.id/target/l1         -> schema.id_column/schema.target_column/schema.l1_column
    - paths.run_dir                -> paths.runs_dir
    - parameters                   -> model_overrides
    - language                     -> defaults.language
    """
    cfg = deepcopy(cfg)

    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("'paths' must be a dictionary.")
    if "run_dir" in paths and "runs_dir" not in paths:
        paths["runs_dir"] = paths["run_dir"]

    if any(k in cfg for k in ("train_file", "dev_file", "test_file")):
        files = cfg.setdefault("files", {})
        if not isinstance(files, dict):
            raise ConfigError("'files' must be a dictionary.")
        if "train_file" in cfg and "train" not in files:
            files["train"] = cfg["train_file"]
        if "dev_file" in cfg and "dev" not in files:
            files["dev"] = cfg["dev_file"]
        if "test_file" in cfg and "test" not in files:
            files["test"] = cfg["test_file"]

    columns = cfg.get("columns", {})
    if columns is None:
        columns = {}
    if not isinstance(columns, dict):
        raise ConfigError("'columns' must be a dictionary if provided.")

    schema = cfg.setdefault("schema", {})
    if not isinstance(schema, dict):
        raise ConfigError("'schema' must be a dictionary.")

    if "id_column" not in schema:
        schema["id_column"] = columns.get("id") or cfg.get("id_column")
    if "target_column" not in schema:
        schema["target_column"] = columns.get("target") or cfg.get("target_column")
    if "l1_column" not in schema:
        schema["l1_column"] = columns.get("l1") or cfg.get("l1_column")

    if "model_name" not in cfg and isinstance(cfg.get("model"), dict):
        model_block = cfg["model"]
        if isinstance(model_block.get("model_name"), str):
            cfg["model_name"] = model_block["model_name"]
        elif isinstance(model_block.get("name"), str):
            cfg["model_name"] = model_block["name"]

    if "model_overrides" not in cfg:
        if isinstance(cfg.get("parameters"), dict):
            cfg["model_overrides"] = deepcopy(cfg["parameters"])
        elif isinstance(cfg.get("model"), dict) and isinstance(cfg["model"].get("parameters"), dict):
            cfg["model_overrides"] = deepcopy(cfg["model"]["parameters"])

    defaults = cfg.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError("'defaults' must be a dictionary.")
    if "language" not in defaults and "language" in cfg:
        defaults["language"] = cfg["language"]

    cfg.setdefault("outputs", {})
    if not isinstance(cfg["outputs"], dict):
        raise ConfigError("'outputs' must be a dictionary.")

    return cfg



def validate_basic_config_structure(cfg: dict[str, Any]) -> None:
    """
    Lightweight structural validation only.
    Detailed semantic validation belongs in validation.py.
    """
    cfg = _ensure_dict(cfg, "Resolved config")

    required_top_level = [
        "paths",
        "schema",
    ]
    for key in required_top_level:
        if key not in cfg:
            raise ConfigError(f"Resolved config is missing required top-level key: '{key}'")

    if "model_name" not in cfg:
        raise ConfigError("Resolved config is missing required top-level key: 'model_name'")

    if "experiment_name" in cfg and not isinstance(cfg["experiment_name"], str):
        raise ConfigError("'experiment_name' must be a string.")

    _ensure_dict(cfg["paths"], "'paths'")
    _ensure_dict(cfg["schema"], "'schema'")

    if "feature_groups" in cfg and not isinstance(cfg["feature_groups"], (list, dict)):
        raise ConfigError("'feature_groups' must be either a list or a dict.")

    for key in ("cv", "runtime", "evaluation", "outputs", "validation", "defaults"):
        if key in cfg and cfg[key] is not None:
            _ensure_dict(cfg[key], f"'{key}'")

    if "model_name" in cfg and not isinstance(cfg["model_name"], str):
        raise ConfigError("'model_name' must be a string.")

    if "model_overrides" in cfg and not isinstance(cfg["model_overrides"], dict):
        raise ConfigError("'model_overrides' must be a dictionary if provided.")



def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Public helper to load a single YAML config without resolving inheritance.
    Useful for debugging or low-level tooling.
    """
    return _read_yaml_file(Path(path))



def load_and_resolve_config(path: str | Path) -> dict[str, Any]:
    """
    Load a config file, recursively resolve inheritance, normalize scaffold keys,
    and perform structural + semantic validation.
    """
    source_path = Path(path).resolve()

    resolved_cfg, lineage = _resolve_inheritance_recursive(
        source_path,
        visited=set(),
        stack=[],
    )
    resolved_cfg = _inject_metadata(
        resolved_cfg,
        source_path=source_path,
        lineage=lineage,
    )
    resolved_cfg = normalize_resolved_config(resolved_cfg)
    validate_basic_config_structure(resolved_cfg)
    resolved_cfg = validate_resolved_config(resolved_cfg)
    return resolved_cfg



def resolve_config(path: str | Path) -> dict[str, Any]:
    """
    Alias for the main config resolution entrypoint.
    """
    return load_and_resolve_config(path)



def save_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    """
    Save the resolved config for reproducibility.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception as e:
        raise ConfigError(f"Failed to save resolved config to {output_path}: {e}") from e
