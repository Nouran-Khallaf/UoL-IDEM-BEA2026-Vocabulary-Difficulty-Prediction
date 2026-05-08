#enhanced_neural_fusion_regressor
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel


_VALID_POOLING = {"cls", "mean", "attention"}


@dataclass(slots=True)
class EnhancedNeuralFusionRegressorConfig:
    encoder_name: str
    tabular_dim: int

    text_pooling: Literal["cls", "mean", "attention"] = "attention"

    tabular_hidden_dim: int = 256
    shared_hidden_dim: int = 256
    fusion_hidden_dim: int = 256

    tabular_num_layers: int = 3
    fusion_num_layers: int = 2

    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    use_layer_norm: bool = True
    freeze_encoder: bool = False

    tabular_use_residual: bool = True
    use_gated_fusion: bool = True
    multi_sample_dropout_samples: int = 5


def _validate_cfg(cfg: EnhancedNeuralFusionRegressorConfig) -> None:
    if not cfg.encoder_name or not cfg.encoder_name.strip():
        raise ValueError("encoder_name must be a non-empty string.")
    if cfg.tabular_dim <= 0:
        raise ValueError("tabular_dim must be > 0.")
    if cfg.text_pooling not in _VALID_POOLING:
        raise ValueError(f"text_pooling must be one of {_VALID_POOLING}.")
    if cfg.tabular_hidden_dim <= 0:
        raise ValueError("tabular_hidden_dim must be > 0.")
    if cfg.shared_hidden_dim <= 0:
        raise ValueError("shared_hidden_dim must be > 0.")
    if cfg.fusion_hidden_dim <= 0:
        raise ValueError("fusion_hidden_dim must be > 0.")
    if cfg.tabular_num_layers <= 0:
        raise ValueError("tabular_num_layers must be > 0.")
    if cfg.fusion_num_layers <= 0:
        raise ValueError("fusion_num_layers must be > 0.")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    if cfg.multi_sample_dropout_samples <= 0:
        raise ValueError("multi_sample_dropout_samples must be > 0.")


def _make_activation(name: str) -> nn.Module:
    resolved = name.strip().lower()
    if resolved == "relu":
        return nn.ReLU()
    if resolved == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dropout: float,
        activation: str,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        return residual + out


class ResidualTabularProjector(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        use_layer_norm: bool,
        use_residual: bool,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

        blocks: list[nn.Module] = []
        n_blocks = max(num_layers - 1, 0)

        for _ in range(n_blocks):
            if use_residual:
                blocks.append(
                    ResidualMLPBlock(
                        hidden_dim,
                        dropout=dropout,
                        activation=activation,
                        use_layer_norm=use_layer_norm,
                    )
                )
            else:
                blocks.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                        _make_activation(activation),
                        nn.Dropout(dropout),
                    ]
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.blocks(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.score(self.dropout(last_hidden_state)).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(last_hidden_state * weights.unsqueeze(-1), dim=1)
        return pooled


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


class EnhancedNeuralFusionRegressor(nn.Module):
    """
    Enhanced transformer + tabular late-fusion regressor with:
    - attention pooling
    - residual tabular branch
    - text/tabular shared projection
    - gated interaction fusion
    - fusion MLP
    - multi-sample dropout
    """

    def __init__(self, cfg: EnhancedNeuralFusionRegressorConfig) -> None:
        super().__init__()
        _validate_cfg(cfg)
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        self.encoder_hidden_size = int(self.encoder.config.hidden_size)

        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.attention_pool = AttentionPooling(
            hidden_size=self.encoder_hidden_size,
            dropout=cfg.dropout,
        )

        self.text_dropout = nn.Dropout(cfg.dropout)

        self.tabular_projector = ResidualTabularProjector(
            input_dim=cfg.tabular_dim,
            hidden_dim=cfg.tabular_hidden_dim,
            num_layers=cfg.tabular_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
            use_residual=cfg.tabular_use_residual,
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, cfg.shared_hidden_dim),
            nn.LayerNorm(cfg.shared_hidden_dim) if cfg.use_layer_norm else nn.Identity(),
            _make_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
        )

        self.tabular_to_shared = nn.Sequential(
            nn.Linear(cfg.tabular_hidden_dim, cfg.shared_hidden_dim),
            nn.LayerNorm(cfg.shared_hidden_dim) if cfg.use_layer_norm else nn.Identity(),
            _make_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
        )

        interaction_dim = cfg.shared_hidden_dim * 4
        self.gate_layer = nn.Linear(interaction_dim, cfg.shared_hidden_dim)

        self.fusion_head = _make_mlp(
            input_dim=interaction_dim + cfg.shared_hidden_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            output_dim=cfg.fusion_hidden_dim,
            num_layers=cfg.fusion_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

        self.multi_sample_dropout = nn.Dropout(cfg.dropout)
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

        if self.cfg.text_pooling == "attention":
            return self.attention_pool(last_hidden_state, attention_mask)

        raise ValueError(f"Unsupported text_pooling '{self.cfg.text_pooling}'.")

    def _apply_multi_sample_dropout(self, hidden: torch.Tensor) -> torch.Tensor:
        preds = []
        for _ in range(self.cfg.multi_sample_dropout_samples):
            dropped = self.multi_sample_dropout(hidden)
            pred = self.output_layer(dropped)
            preds.append(pred)
        return torch.mean(torch.stack(preds, dim=0), dim=0)

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
        text_repr = self.text_dropout(text_repr)

        tabular_repr = self.tabular_projector(tabular_features)

        text_shared = self.text_projector(text_repr)
        tab_shared = self.tabular_to_shared(tabular_repr)

        abs_diff = torch.abs(text_shared - tab_shared)
        elem_prod = text_shared * tab_shared

        interaction = torch.cat(
            [text_shared, tab_shared, abs_diff, elem_prod],
            dim=-1,
        )

        if self.cfg.use_gated_fusion:
            gate = torch.sigmoid(self.gate_layer(interaction))
            gated_mix = gate * text_shared + (1.0 - gate) * tab_shared
        else:
            gated_mix = 0.5 * (text_shared + tab_shared)

        fused = torch.cat([interaction, gated_mix], dim=-1)
        hidden = self.fusion_head(fused)

        output = self._apply_multi_sample_dropout(hidden).squeeze(-1)
        return output