from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel


_VALID_POOLING = {"cls", "mean"}


@dataclass(slots=True)
class NeuralFusionRegressorConfig:
    """
    Configuration for transformer + tabular late-fusion regression.

    Architecture
    ------------
    text encoder -> pooled text vector
                 -> tabular projection MLP
                 -> concatenation
                 -> fusion MLP
                 -> scalar regression output

    Notes
    -----
    - This is the recommended 'second architecture' for combining
      transformer text representations with engineered features.
    - It supports multilingual encoders such as:
        - bert-base-multilingual-cased
        - xlm-roberta-base
        - sentence-transformers/LaBSE
    - Tabular features are injected after text encoding, which is much
      cleaner than forcing them into the token sequence.
    """
    encoder_name: str
    tabular_dim: int
    text_pooling: Literal["cls", "mean"] = "cls"
    tabular_hidden_dim: int = 256
    fusion_hidden_dim: int = 256
    tabular_num_layers: int = 2
    fusion_num_layers: int = 2
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    use_layer_norm: bool = True
    freeze_encoder: bool = False


def _validate_cfg(cfg: NeuralFusionRegressorConfig) -> None:
    if not cfg.encoder_name or not cfg.encoder_name.strip():
        raise ValueError("encoder_name must be a non-empty string.")
    if cfg.tabular_dim <= 0:
        raise ValueError("tabular_dim must be > 0.")
    if cfg.text_pooling not in _VALID_POOLING:
        raise ValueError(f"text_pooling must be one of {_VALID_POOLING}.")
    if cfg.tabular_hidden_dim <= 0:
        raise ValueError("tabular_hidden_dim must be > 0.")
    if cfg.fusion_hidden_dim <= 0:
        raise ValueError("fusion_hidden_dim must be > 0.")
    if cfg.tabular_num_layers <= 0:
        raise ValueError("tabular_num_layers must be > 0.")
    if cfg.fusion_num_layers <= 0:
        raise ValueError("fusion_num_layers must be > 0.")
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
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        return nn.Sequential(*layers)

    layers = []
    in_dim = input_dim
    act = _make_activation(activation)

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act.__class__())
        layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class NeuralFusionRegressor(nn.Module):
    """
    Transformer + tabular late-fusion regressor.
    """

    def __init__(self, cfg: NeuralFusionRegressorConfig) -> None:
        super().__init__()
        _validate_cfg(cfg)
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        self.encoder_hidden_size = int(self.encoder.config.hidden_size)

        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.tabular_projector = _make_mlp(
            input_dim=cfg.tabular_dim,
            hidden_dim=cfg.tabular_hidden_dim,
            output_dim=cfg.tabular_hidden_dim,
            num_layers=cfg.tabular_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

        fusion_input_dim = self.encoder_hidden_size + cfg.tabular_hidden_dim

        self.fusion_head = _make_mlp(
            input_dim=fusion_input_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            output_dim=cfg.fusion_hidden_dim,
            num_layers=cfg.fusion_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

        self.output_layer = nn.Linear(cfg.fusion_hidden_dim, 1)

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
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_repr = self._pool_text(
            last_hidden_state=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

        tabular_repr = self.tabular_projector(tabular_features)
        fused = torch.cat([text_repr, tabular_repr], dim=-1)
        hidden = self.fusion_head(fused)
        output = self.output_layer(hidden).squeeze(-1)

        return output