from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel


_VALID_POOLING = {"cls", "mean"}


@dataclass(slots=True)
class TextFeatureRegressorConfig:
    """
    Configuration for prompt-based transformer regression.

    Architecture
    ------------
    prompt text -> transformer encoder -> pooled text representation
                -> regression MLP head -> scalar score

    Notes
    -----
    - This is the model for the third architecture:
      engineered features appended as structured text.
    - It is suitable for multilingual encoders such as:
        - bert-base-multilingual-cased
        - xlm-roberta-base
        - sentence-transformers/LaBSE
    - Compared with the neural fusion architecture, this model has no separate
      tabular branch; all engineered features are verbalized into the input text.
    """
    encoder_name: str
    text_pooling: Literal["cls", "mean"] = "cls"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    use_layer_norm: bool = True
    freeze_encoder: bool = False


def _validate_cfg(cfg: TextFeatureRegressorConfig) -> None:
    if not cfg.encoder_name or not cfg.encoder_name.strip():
        raise ValueError("encoder_name must be a non-empty string.")
    if cfg.text_pooling not in _VALID_POOLING:
        raise ValueError(f"text_pooling must be one of {_VALID_POOLING}.")
    if cfg.hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0.")
    if cfg.num_layers <= 0:
        raise ValueError("num_layers must be > 0.")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError("dropout must be in [0, 1).")


def _make_activation(name: str) -> nn.Module:
    resolved = name.strip().lower()
    if resolved == "relu":
        return nn.ReLU()
    if resolved == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _make_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
    use_layer_norm: bool,
) -> nn.Sequential:
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = []
    in_dim = input_dim

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_make_activation(activation))
        layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class TextFeatureRegressor(nn.Module):
    """
    Prompt-based transformer regressor.

    Input
    -----
    A single tokenized prompt sequence that already includes:
    - main text/context
    - engineered features rendered as text

    Output
    ------
    One scalar regression prediction per instance.
    """

    def __init__(self, cfg: TextFeatureRegressorConfig) -> None:
        super().__init__()
        _validate_cfg(cfg)
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        self.encoder_hidden_size = int(self.encoder.config.hidden_size)

        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.regression_head = _make_mlp(
            input_dim=self.encoder_hidden_size,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

        self.output_layer = nn.Linear(cfg.hidden_dim, 1)

    def _pool_text(
        self,
        *,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.text_pooling == "cls":
            return last_hidden_state[:, 0, :]

        if self.cfg.text_pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        raise ValueError(f"Unsupported text_pooling '{self.cfg.text_pooling}'.")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self._pool_text(
            last_hidden_state=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

        hidden = self.regression_head(pooled)
        output = self.output_layer(hidden).squeeze(-1)
        return output