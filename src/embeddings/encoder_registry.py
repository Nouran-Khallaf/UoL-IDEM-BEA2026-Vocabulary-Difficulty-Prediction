from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EncoderSpec:
    """
    Lightweight registry entry for a text encoder used in late-fusion
    embedding generation.

    Attributes
    ----------
    key:
        Short internal key used in configs, e.g. 'mbert', 'xlmr', 'labse'.
    hf_name:
        Hugging Face model identifier.
    default_prefix:
        Prefix used for generated embedding columns.
    default_pooling:
        Recommended pooling strategy for this encoder.
    max_length:
        Recommended default max token length.
    description:
        Human-readable note for debugging / metadata.
    """
    key: str
    hf_name: str
    default_prefix: str
    default_pooling: str = "cls"
    max_length: int = 128
    description: str = ""


_ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "mbert": EncoderSpec(
        key="mbert",
        hf_name="bert-base-multilingual-cased",
        default_prefix="mbert",
        default_pooling="cls",
        max_length=128,
        description="Multilingual BERT base, cased.",
    ),
    "xlmr_base": EncoderSpec(
        key="xlmr_base",
        hf_name="xlm-roberta-base",
        default_prefix="xlmr",
        default_pooling="cls",
        max_length=128,
        description="XLM-RoBERTa base multilingual encoder.",
    ),
    "xlmr_large": EncoderSpec(
        key="xlmr_large",
        hf_name="xlm-roberta-large",
        default_prefix="xlmr_large",
        default_pooling="cls",
        max_length=128,
        description="XLM-RoBERTa large multilingual encoder.",
    ),
    "labse": EncoderSpec(
        key="labse",
        hf_name="sentence-transformers/LaBSE",
        default_prefix="labse",
        default_pooling="cls",
        max_length=128,
        description="LaBSE multilingual sentence embedding model.",
    ),
}


def list_supported_encoders() -> list[str]:
    return sorted(_ENCODER_REGISTRY.keys())


def has_encoder(key: str) -> bool:
    return key.strip().lower() in _ENCODER_REGISTRY


def get_encoder_spec(key: str) -> EncoderSpec:
    resolved = key.strip().lower()
    if resolved not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unsupported encoder key '{key}'. "
            f"Supported encoders: {list_supported_encoders()}"
        )
    return _ENCODER_REGISTRY[resolved]


def resolve_encoder_params(
    *,
    encoder_key: str | None = None,
    encoder_name: str | None = None,
    prefix: str | None = None,
    pooling: str | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """
    Resolve a config-friendly encoder specification.

    Usage patterns
    --------------
    1. Registry-based:
       resolve_encoder_params(encoder_key="labse")

    2. Explicit Hugging Face name:
       resolve_encoder_params(
           encoder_name="xlm-roberta-base",
           prefix="xlmr",
           pooling="cls",
           max_length=128,
       )

    Returns
    -------
    dict[str, Any]
        Flat parameter dictionary suitable for embedding extraction utilities.
    """
    if encoder_key is not None:
        spec = get_encoder_spec(encoder_key)
        return {
            "encoder_key": spec.key,
            "encoder_name": spec.hf_name,
            "prefix": prefix or spec.default_prefix,
            "pooling": pooling or spec.default_pooling,
            "max_length": int(max_length or spec.max_length),
            "description": spec.description,
        }

    if encoder_name is None or not encoder_name.strip():
        raise ValueError(
            "Either encoder_key or encoder_name must be provided."
        )

    resolved_prefix = (prefix or "emb").strip()
    resolved_pooling = (pooling or "cls").strip().lower()
    resolved_max_length = int(max_length or 128)

    return {
        "encoder_key": None,
        "encoder_name": encoder_name.strip(),
        "prefix": resolved_prefix,
        "pooling": resolved_pooling,
        "max_length": resolved_max_length,
        "description": "Explicit user-supplied encoder.",
    }