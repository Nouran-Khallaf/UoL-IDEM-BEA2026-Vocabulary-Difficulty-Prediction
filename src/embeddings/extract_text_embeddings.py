from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


_VALID_POOLING = {"cls", "mean"}


@dataclass(slots=True)
class TextEmbeddingConfig:
    """
    Configuration for transformer-based embedding extraction.

    Notes
    -----
    - The extractor is model-agnostic and can work with BERT, mBERT,
      XLM-R, or LaBSE-style encoders as long as they are loadable through
      Hugging Face AutoTokenizer / AutoModel.
    - Text fields are concatenated in a controlled way with a separator token
      string so the downstream representation can jointly encode context,
      source word, clue, target word, etc.
    - Output is a dense dataframe with columns:
        {prefix}_0, {prefix}_1, ..., {prefix}_{hidden_size-1}
    """
    encoder_name: str
    text_columns: Sequence[str]
    prefix: str = "emb"
    batch_size: int = 16
    max_length: int = 128
    pooling: str = "cls"
    device: str | None = None
    add_special_tokens: bool = True
    separator_text: str = " [SEP] "


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_pooling(pooling: str) -> str:
    resolved = pooling.strip().lower()
    if resolved not in _VALID_POOLING:
        raise ValueError(
            f"Unsupported pooling='{pooling}'. Expected one of: {sorted(_VALID_POOLING)}"
        )
    return resolved


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


class TextEmbeddingExtractor:
    """
    Transformer embedding extractor for tabular NLP pipelines.
    """

    def __init__(self, cfg: TextEmbeddingConfig) -> None:
        if not cfg.encoder_name or not cfg.encoder_name.strip():
            raise ValueError("encoder_name must be a non-empty string.")
        if not cfg.text_columns:
            raise ValueError("text_columns must be a non-empty sequence.")
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if cfg.max_length <= 0:
            raise ValueError("max_length must be > 0.")

        self.cfg = cfg
        self.cfg = TextEmbeddingConfig(
            encoder_name=cfg.encoder_name,
            text_columns=list(cfg.text_columns),
            prefix=cfg.prefix,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            pooling=_validate_pooling(cfg.pooling),
            device=_resolve_device(cfg.device),
            add_special_tokens=cfg.add_special_tokens,
            separator_text=cfg.separator_text,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.encoder_name)
        self.model = AutoModel.from_pretrained(self.cfg.encoder_name)
        self.model.to(self.cfg.device)
        self.model.eval()

    def _build_texts(self, df: pd.DataFrame) -> list[str]:
        missing = [c for c in self.cfg.text_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing text columns for embedding extraction: {missing}"
            )

        texts: list[str] = []
        for _, row in df.iterrows():
            parts = [_safe_text(row[col]) for col in self.cfg.text_columns]
            text = self.cfg.separator_text.join(parts).strip()
            texts.append(text)
        return texts

    def _pool(
        self,
        *,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return last_hidden_state[:, 0, :]

        if self.cfg.pooling == "mean":
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * expanded_mask, dim=1)
            counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
            return summed / counts

        raise ValueError(f"Unsupported pooling mode: {self.cfg.pooling}")

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty.")

        all_vectors: list[np.ndarray] = []

        for start in range(0, len(texts), self.cfg.batch_size):
            batch_texts = list(texts[start : start + self.cfg.batch_size])

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                add_special_tokens=self.cfg.add_special_tokens,
                return_tensors="pt",
            )
            enc = {k: v.to(self.cfg.device) for k, v in enc.items()}

            outputs = self.model(**enc)
            pooled = self._pool(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=enc["attention_mask"],
            )
            all_vectors.append(pooled.detach().cpu().numpy())

        matrix = np.vstack(all_vectors)
        return matrix

    @torch.no_grad()
    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = self._build_texts(df)
        matrix = self.encode_texts(texts)

        columns = [f"{self.cfg.prefix}_{i}" for i in range(matrix.shape[1])]
        emb_df = pd.DataFrame(matrix, columns=columns, index=df.index)
        return emb_df


def augment_dataframe_with_embeddings(
    df: pd.DataFrame,
    *,
    encoder_name: str,
    text_columns: Sequence[str],
    prefix: str = "emb",
    batch_size: int = 16,
    max_length: int = 128,
    pooling: str = "cls",
    device: str | None = None,
    add_special_tokens: bool = True,
    separator_text: str = " [SEP] ",
) -> pd.DataFrame:
    """
    Return a new dataframe with embedding columns appended.

    Parameters
    ----------
    df:
        Input dataframe.
    encoder_name:
        Hugging Face model name, e.g.:
        - bert-base-multilingual-cased
        - xlm-roberta-base
        - sentence-transformers/LaBSE
    text_columns:
        Text columns to concatenate and encode.
    prefix:
        Prefix for generated embedding feature columns.
    batch_size:
        Inference batch size.
    max_length:
        Max token length for truncation.
    pooling:
        'cls' or 'mean'
    device:
        'cpu', 'cuda', etc. If None, auto-detect.
    add_special_tokens:
        Passed to tokenizer.
    separator_text:
        String inserted between text fields.

    Returns
    -------
    pd.DataFrame
        Original dataframe plus embedding columns.
    """
    cfg = TextEmbeddingConfig(
        encoder_name=encoder_name,
        text_columns=list(text_columns),
        prefix=prefix,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        device=device,
        add_special_tokens=add_special_tokens,
        separator_text=separator_text,
    )
    extractor = TextEmbeddingExtractor(cfg)
    emb_df = extractor.encode_dataframe(df)

    return pd.concat(
        [
            df.reset_index(drop=True),
            emb_df.reset_index(drop=True),
        ],
        axis=1,
    )


def save_embedding_augmented_csv(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    encoder_name: str,
    text_columns: Sequence[str],
    prefix: str = "emb",
    batch_size: int = 16,
    max_length: int = 128,
    pooling: str = "cls",
    device: str | None = None,
    add_special_tokens: bool = True,
    separator_text: str = " [SEP] ",
) -> None:
    """
    Read a CSV, append transformer embeddings, and save the result.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    out_df = augment_dataframe_with_embeddings(
        df,
        encoder_name=encoder_name,
        text_columns=text_columns,
        prefix=prefix,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        device=device,
        add_special_tokens=add_special_tokens,
        separator_text=separator_text,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


def infer_embedding_column_names(
    *,
    hidden_size: int,
    prefix: str,
) -> list[str]:
    """
    Utility for downstream config/diagnostic generation.
    """
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0.")
    if not prefix or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    return [f"{prefix}_{i}" for i in range(hidden_size)]