#enhanced_neural_fusion_regressor
from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path
from typing import Any, Literal
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

    Explainability-ready additions:
    - encode_text_shared()
    - encode_tabular_shared()
    - fuse_shared()
    - predict_from_shared()
    - forward_with_intermediates()
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

    # ------------------------------------------------------------------
    # Explainability-ready branch accessors
    # ------------------------------------------------------------------
    def encode_text_shared(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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
        text_shared = self.text_projector(text_repr)
        return text_shared

    def encode_tabular_shared(
        self,
        *,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        tabular_repr = self.tabular_projector(tabular_features)
        tab_shared = self.tabular_to_shared(tabular_repr)
        return tab_shared

    def fuse_shared(
        self,
        *,
        text_shared: torch.Tensor,
        tab_shared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return hidden, interaction, gated_mix

    def predict_from_shared(
        self,
        *,
        text_shared: torch.Tensor,
        tab_shared: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _, _ = self.fuse_shared(
            text_shared=text_shared,
            tab_shared=tab_shared,
        )
        output = self._apply_multi_sample_dropout(hidden).squeeze(-1)
        return output

    def forward_with_intermediates(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_shared = self.encode_text_shared(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tab_shared = self.encode_tabular_shared(
            tabular_features=tabular_features,
        )
        hidden, interaction, gated_mix = self.fuse_shared(
            text_shared=text_shared,
            tab_shared=tab_shared,
        )
        output = self._apply_multi_sample_dropout(hidden).squeeze(-1)

        return {
            "prediction": output,
            "text_shared": text_shared,
            "tab_shared": tab_shared,
            "interaction": interaction,
            "gated_mix": gated_mix,
            "hidden": hidden,
        }

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        text_shared = self.encode_text_shared(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tab_shared = self.encode_tabular_shared(
            tabular_features=tabular_features,
        )
        return self.predict_from_shared(
            text_shared=text_shared,
            tab_shared=tab_shared,
        )


class FixedTextTabularExpectedGradientsModel(nn.Module):
    """
    SHAP / IG-style wrapper:
    fixes one text context and explains the tabular branch only.

    This is the practical way to get feature-level explanations for the
    engineered tabular inputs while conditioning on the actual text inputs.
    """

    def __init__(
        self,
        base_model: EnhancedNeuralFusionRegressor,
        *,
        fixed_input_ids: torch.Tensor,
        fixed_attention_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        with torch.no_grad():
            fixed_text_shared = self.base_model.encode_text_shared(
                input_ids=fixed_input_ids,
                attention_mask=fixed_attention_mask,
            )

        self.register_buffer("fixed_text_shared", fixed_text_shared.detach())

    def forward(self, tabular_features: torch.Tensor) -> torch.Tensor:
        batch_size = int(tabular_features.shape[0])
        text_shared = self.fixed_text_shared.expand(batch_size, -1)
        tab_shared = self.base_model.encode_tabular_shared(
            tabular_features=tabular_features,
        )
        return self.base_model.predict_from_shared(
            text_shared=text_shared,
            tab_shared=tab_shared,
        )


def _normalize_shap_values(shap_values: Any) -> Any:
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            raise ValueError("Expected a single-output regression SHAP result.")
    return shap_values


def run_tabular_expected_gradients(
    *,
    model: EnhancedNeuralFusionRegressor,
    explain_tabular: torch.Tensor,
    background_tabular: torch.Tensor,
    explain_input_ids: torch.Tensor,
    explain_attention_mask: torch.Tensor,
    feature_names: list[str],
    output_dir: Path | None = None,
    explain_item_ids: list[str] | None = None,
    max_display: int = 20,
    dependence_top_k: int = 8,
) -> dict[str, Any]:
    """
    Runs SHAP GradientExplainer (expected-gradients / IG-style explanation)
    for the tabular branch under fixed text contexts.

    Each explained row uses its own text context and its own wrapper.
    This is heavier than tree SHAP, but much more faithful to the actual
    neural fusion model.
    """
    import json
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt

    model.eval()
    device = next(model.parameters()).device

    explain_tabular = explain_tabular.to(device)
    background_tabular = background_tabular.to(device)
    explain_input_ids = explain_input_ids.to(device)
    explain_attention_mask = explain_attention_mask.to(device)

    if explain_tabular.ndim != 2:
        raise ValueError("explain_tabular must have shape [n_rows, n_features].")
    if background_tabular.ndim != 2:
        raise ValueError("background_tabular must have shape [n_rows, n_features].")
    if explain_input_ids.size(0) != explain_tabular.size(0):
        raise ValueError("explain_input_ids and explain_tabular must have the same number of rows.")
    if explain_attention_mask.size(0) != explain_tabular.size(0):
        raise ValueError("explain_attention_mask and explain_tabular must have the same number of rows.")
    if explain_tabular.size(1) != len(feature_names):
        raise ValueError("Number of feature names does not match explain_tabular.shape[1].")

    shap_rows = []
    base_rows = []
    pred_rows = []

    for i in range(explain_tabular.size(0)):
        row_model = FixedTextTabularExpectedGradientsModel(
            model,
            fixed_input_ids=explain_input_ids[i : i + 1],
            fixed_attention_mask=explain_attention_mask[i : i + 1],
        ).to(device)
        row_model.eval()

        explainer = shap.GradientExplainer(row_model, background_tabular)

        row_x = explain_tabular[i : i + 1]
        row_shap = explainer.shap_values(row_x)
        row_shap = _normalize_shap_values(row_shap)

        if hasattr(row_shap, "detach"):
            row_shap = row_shap.detach().cpu().numpy()
        else:
            row_shap = np.asarray(row_shap)

        if row_shap.ndim == 3 and row_shap.shape[-1] == 1:
            row_shap = row_shap[..., 0]
        if row_shap.ndim == 1:
            row_shap = row_shap[None, :]
        if row_shap.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape for row {i}: {row_shap.shape}")

        row_pred = row_model(row_x).detach().cpu().numpy().reshape(-1)
        row_base = row_pred - row_shap.sum(axis=1)

        shap_rows.append(row_shap[0])
        pred_rows.append(float(row_pred[0]))
        base_rows.append(float(row_base[0]))

    values = np.vstack(shap_rows)
    data = explain_tabular.detach().cpu().numpy()
    base_values = np.asarray(base_rows, dtype=np.float32)
    preds = np.asarray(pred_rows, dtype=np.float32)

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
    )

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values(["mean_abs_shap", "feature_name"], ascending=[False, True]).reset_index(drop=True)

    result = {
        "explanation": explanation,
        "importance_df": importance_df,
        "predictions": preds,
        "base_values": base_values,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        importance_df.to_csv(output_dir / "tabular_expected_gradients_importance.csv", index=False)

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "tabular_expected_gradients_beeswarm.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.plots.bar(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "tabular_expected_gradients_bar.png", dpi=220, bbox_inches="tight")
        plt.close()

        top_features = importance_df["feature_name"].head(dependence_top_k).tolist()
        for feature_name in top_features:
            safe_name = feature_name.replace("/", "_")
            plt.figure(figsize=(8, 6))
            shap.plots.scatter(explanation[:, feature_name], color=explanation, show=False)
            plt.tight_layout()
            plt.savefig(
                output_dir / f"tabular_expected_gradients_scatter_{safe_name}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close()

        metadata = {
            "n_explained_rows": int(explain_tabular.size(0)),
            "n_background_rows": int(background_tabular.size(0)),
            "n_features": int(explain_tabular.size(1)),
            "top_global_feature": None if importance_df.empty else str(importance_df.iloc[0]["feature_name"]),
            "explanation_mode": "SHAP GradientExplainer expected-gradients (IG-style), tabular branch conditioned on fixed text",
        }
        with (output_dir / "tabular_expected_gradients_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if explain_item_ids is not None:
            local_records = []
            for row_idx, item_id in enumerate(explain_item_ids):
                order = np.argsort(np.abs(values[row_idx]))[::-1]
                local_records.append(
                    {
                        "item_id": str(item_id),
                        "prediction": float(preds[row_idx]),
                        "base_value": float(base_values[row_idx]),
                        "top_contributions": [
                            {
                                "feature_name": feature_names[j],
                                "feature_value": float(data[row_idx, j]),
                                "shap_value": float(values[row_idx, j]),
                                "abs_shap_value": float(abs(values[row_idx, j])),
                            }
                            for j in order[:20]
                        ],
                    }
                )
            with (output_dir / "tabular_expected_gradients_local.json").open("w", encoding="utf-8") as f:
                json.dump({"local_explanations": local_records}, f, indent=2, ensure_ascii=False)

    return result