from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.exceptions import ConfigError
from src.embeddings.encoder_registry import resolve_encoder_params


@dataclass(slots=True)
class LateFusionConfig:
    """
    Resolved late-fusion experiment settings.

    Attributes
    ----------
    enabled:
        Whether late fusion is enabled.
    encoder_key:
        Short registry key, e.g. 'labse', 'mbert', 'xlmr_base'.
    encoder_name:
        Full Hugging Face model identifier.
    prefix:
        Embedding feature prefix used in generated CSV columns.
    pooling:
        Pooling strategy used during embedding extraction.
    max_length:
        Maximum tokenizer sequence length.
    text_columns:
        Text columns used to build the embedding input.
    feature_dir:
        Directory containing processed feature CSV/JSON files.
    train_file:
        Embedding-augmented train CSV filename.
    dev_file:
        Embedding-augmented dev CSV filename.
    train_diagnostics_file:
        Matching train diagnostics JSON filename.
    dev_diagnostics_file:
        Matching dev diagnostics JSON filename.
    """
    enabled: bool
    encoder_key: str | None
    encoder_name: str | None
    prefix: str | None
    pooling: str | None
    max_length: int | None
    text_columns: list[str]
    feature_dir: Path | None
    train_file: str | None
    dev_file: str | None
    train_diagnostics_file: str | None
    dev_diagnostics_file: str | None


def _as_dict(cfg: Any, name: str) -> dict[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"'{name}' must be a dictionary.")
    return cfg


def _default_augmented_name(base_filename: str, suffix: str) -> str:
    path = Path(base_filename)
    return f"{path.stem}_{suffix}{path.suffix}"


def _resolve_feature_dir(resolved_config: dict[str, Any]) -> Path:
    tabular_input = _as_dict(resolved_config.get("tabular_input"), "tabular_input")
    if "feature_dir" in tabular_input:
        return Path(str(tabular_input["feature_dir"])).resolve()

    paths = _as_dict(resolved_config.get("paths"), "paths")
    processed_dir = paths.get("processed_data_dir", "data/processed")
    experiment_name = resolved_config.get("experiment_name", "feature_build")
    return (Path(processed_dir) / str(experiment_name)).resolve()


def _resolve_base_filenames(resolved_config: dict[str, Any]) -> tuple[str, str, str, str]:
    tabular_input = _as_dict(resolved_config.get("tabular_input"), "tabular_input")

    train_file = str(tabular_input.get("train_file", "train_features.csv"))
    dev_file = str(tabular_input.get("dev_file", "dev_features.csv"))
    train_diag = str(tabular_input.get("train_diagnostics_file", "train_feature_diagnostics.json"))
    dev_diag = str(tabular_input.get("dev_diagnostics_file", "dev_feature_diagnostics.json"))

    return train_file, dev_file, train_diag, dev_diag


def resolve_late_fusion_config(resolved_config: dict[str, Any]) -> LateFusionConfig:
    """
    Resolve late-fusion settings from the experiment YAML/config.

    Expected config pattern
    -----------------------
    fusion:
      enabled: true
      mode: late_fusion
      encoder_key: labse
      # OR encoder_name: sentence-transformers/LaBSE
      prefix: null
      pooling: null
      max_length: null
      text_columns:
        - L1_context
        - L1_source_word
        - en_target_word

    tabular_input:
      feature_dir: data/processed/de_features_v1
      train_file: train_features.csv
      dev_file: dev_features.csv
      train_diagnostics_file: train_feature_diagnostics.json
      dev_diagnostics_file: dev_feature_diagnostics.json
    """
    fusion = _as_dict(resolved_config.get("fusion"), "fusion")
    enabled = bool(fusion.get("enabled", False))
    mode = str(fusion.get("mode", "")).strip().lower()

    if not enabled:
        return LateFusionConfig(
            enabled=False,
            encoder_key=None,
            encoder_name=None,
            prefix=None,
            pooling=None,
            max_length=None,
            text_columns=[],
            feature_dir=None,
            train_file=None,
            dev_file=None,
            train_diagnostics_file=None,
            dev_diagnostics_file=None,
        )

    if mode not in {"late_fusion", "embeddings_tabular", "embedding_tabular"}:
        raise ConfigError(
            "fusion.enabled is true, but fusion.mode must be one of "
            "{'late_fusion', 'embeddings_tabular', 'embedding_tabular'}."
        )

    text_columns = fusion.get("text_columns", [])
    if not isinstance(text_columns, list) or not text_columns:
        raise ConfigError(
            "Late-fusion config requires fusion.text_columns as a non-empty list."
        )
    text_columns = [str(c).strip() for c in text_columns if str(c).strip()]
    if not text_columns:
        raise ConfigError("Late-fusion text_columns resolved to an empty list.")

    encoder_key = fusion.get("encoder_key")
    encoder_name = fusion.get("encoder_name")
    prefix = fusion.get("prefix")
    pooling = fusion.get("pooling")
    max_length = fusion.get("max_length")

    encoder_params = resolve_encoder_params(
        encoder_key=encoder_key,
        encoder_name=encoder_name,
        prefix=prefix,
        pooling=pooling,
        max_length=max_length,
    )

    resolved_prefix = str(encoder_params["prefix"])
    resolved_encoder_name = str(encoder_params["encoder_name"])
    resolved_pooling = str(encoder_params["pooling"])
    resolved_max_length = int(encoder_params["max_length"])
    resolved_encoder_key = encoder_params["encoder_key"]

    feature_dir = _resolve_feature_dir(resolved_config)
    base_train_file, base_dev_file, base_train_diag, base_dev_diag = _resolve_base_filenames(resolved_config)

    augmented_train_file = str(
        fusion.get("train_file") or _default_augmented_name(base_train_file, resolved_prefix)
    )
    augmented_dev_file = str(
        fusion.get("dev_file") or _default_augmented_name(base_dev_file, resolved_prefix)
    )
    augmented_train_diag = str(
        fusion.get("train_diagnostics_file") or _default_augmented_name(base_train_diag, resolved_prefix)
    )
    augmented_dev_diag = str(
        fusion.get("dev_diagnostics_file") or _default_augmented_name(base_dev_diag, resolved_prefix)
    )

    return LateFusionConfig(
        enabled=True,
        encoder_key=resolved_encoder_key,
        encoder_name=resolved_encoder_name,
        prefix=resolved_prefix,
        pooling=resolved_pooling,
        max_length=resolved_max_length,
        text_columns=text_columns,
        feature_dir=feature_dir,
        train_file=augmented_train_file,
        dev_file=augmented_dev_file,
        train_diagnostics_file=augmented_train_diag,
        dev_diagnostics_file=augmented_dev_diag,
    )


def apply_late_fusion_tabular_overrides(
    resolved_config: dict[str, Any],
    late_fusion_cfg: LateFusionConfig,
) -> dict[str, Any]:
    """
    Return a modified config where tabular_input points to the
    embedding-augmented feature files.

    This lets the existing tabular runner consume late-fusion features without
    any structural changes to the training loop.
    """
    if not late_fusion_cfg.enabled:
        return resolved_config

    cfg = dict(resolved_config)
    tabular_input = dict(_as_dict(cfg.get("tabular_input"), "tabular_input"))

    if late_fusion_cfg.feature_dir is None:
        raise ConfigError("Late-fusion feature_dir is missing.")
    if late_fusion_cfg.train_file is None or late_fusion_cfg.train_diagnostics_file is None:
        raise ConfigError("Late-fusion train file/diagnostics are missing.")

    tabular_input["feature_dir"] = str(late_fusion_cfg.feature_dir)
    tabular_input["train_file"] = late_fusion_cfg.train_file
    tabular_input["train_diagnostics_file"] = late_fusion_cfg.train_diagnostics_file

    if late_fusion_cfg.dev_file is not None:
        tabular_input["dev_file"] = late_fusion_cfg.dev_file
    if late_fusion_cfg.dev_diagnostics_file is not None:
        tabular_input["dev_diagnostics_file"] = late_fusion_cfg.dev_diagnostics_file

    cfg["tabular_input"] = tabular_input

    metadata = dict(_as_dict(cfg.get("resolved_metadata"), "resolved_metadata"))
    metadata["late_fusion"] = {
        "enabled": late_fusion_cfg.enabled,
        "encoder_key": late_fusion_cfg.encoder_key,
        "encoder_name": late_fusion_cfg.encoder_name,
        "prefix": late_fusion_cfg.prefix,
        "pooling": late_fusion_cfg.pooling,
        "max_length": late_fusion_cfg.max_length,
        "text_columns": late_fusion_cfg.text_columns,
        "feature_dir": None if late_fusion_cfg.feature_dir is None else str(late_fusion_cfg.feature_dir),
        "train_file": late_fusion_cfg.train_file,
        "dev_file": late_fusion_cfg.dev_file,
        "train_diagnostics_file": late_fusion_cfg.train_diagnostics_file,
        "dev_diagnostics_file": late_fusion_cfg.dev_diagnostics_file,
    }
    cfg["resolved_metadata"] = metadata

    return cfg