from __future__ import annotations

import argparse
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import yaml
from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# What this script does
# -----------------------------------------------------------------------------
# 1) Loads a saved neural-fusion run directory:
#       - final_model.pt
#       - resolved_config.yaml
#       - tabular_preprocessor.pkl
# 2) Reloads the dev feature table referenced by the config.
# 3) Rebuilds the model via a project-specific builder function.
# 4) Wraps the neural model so SHAP explains ONLY the tabular numeric branch,
#    while the text branch stays fixed for each explained batch.
# 5) Runs GradientExplainer on a sampled dev subset.
# 6) Saves:
#       - mean absolute SHAP importances
#       - beeswarm plot (colored by feature value)
#       - bar plot
#       - colored dependence/scatter plots
#       - optional local explanation JSON for selected item_ids
#
# IMPORTANT:
# You must point --builder at a function in your codebase that returns the
# instantiated model for this config, e.g.
#   mypkg.models.neural_fusion:build_model_from_config
# or
#   mypkg.training.model_factory:build_model
#
# The builder function must accept at least:
#   builder(config: dict[str, Any]) -> torch.nn.Module
# -----------------------------------------------------------------------------


@dataclass
class RunArtifacts:
    run_dir: Path
    config: dict[str, Any]
    model_path: Path
    preprocessor_path: Path
    dev_csv_path: Path
    target_column: str
    id_column: str
    text_columns: list[str]
    feature_columns: list[str]


class TabularOnlyWrapper(torch.nn.Module):
    """
    A SHAP-facing wrapper that explains only the tabular tensor while keeping
    the text tensors fixed for the current batch.

    The wrapped base model must support one of these forward signatures:

      1) model(input_ids=..., attention_mask=..., tabular_features=...)
      2) model(batch_dict)

    and must return either:
      - a tensor of shape [batch] or [batch, 1]
      - a dict containing one of: 'preds', 'prediction', 'logits', 'output'

    If your project uses different names, edit `_run_model_forward` below.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        fixed_inputs: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.fixed_inputs = fixed_inputs
        self.device = device

    def forward(self, tabular_features: torch.Tensor) -> torch.Tensor:
        return _run_model_forward(
            self.base_model,
            fixed_inputs=self.fixed_inputs,
            tabular_features=tabular_features,
            device=self.device,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SHAP directly on the saved neural fusion model, explaining only "
            "the tabular feature branch while keeping text inputs fixed."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing final_model.pt, resolved_config.yaml, and tabular_preprocessor.pkl.",
    )
    parser.add_argument(
        "--builder",
        type=str,
        required=True,
        help=(
            "Project-specific model builder in the form module.submodule:function, "
            "for example bea_kvl_runs.src.models.neural_fusion:build_model_from_config"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for SHAP outputs. Defaults to <run-dir>/shap_tabular.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Optional project root used to resolve relative paths in resolved_config.yaml. "
            "Defaults to the parent of the runs directory when possible, otherwise cwd."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference and SHAP. Example: cuda or cpu.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=64,
        help="Number of dev rows used as the SHAP background set.",
    )
    parser.add_argument(
        "--explain-size",
        type=int,
        default=256,
        help="Number of dev rows to explain.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for tokenizer inference and SHAP evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for row sampling.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum number of features shown in summary plots.",
    )
    parser.add_argument(
        "--dependence-top-k",
        type=int,
        default=8,
        help="Number of top features for which to save colored dependence plots.",
    )
    parser.add_argument(
        "--local-item-ids",
        type=str,
        default="",
        help="Optional comma-separated item_id values for per-instance local explanation export.",
    )
    parser.add_argument(
        "--strict-state-dict",
        action="store_true",
        help="Load the checkpoint strictly. By default strict=False is used for convenience.",
    )
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_project_root(run_dir: Path, explicit_project_root: Path | None) -> Path:
    if explicit_project_root is not None:
        return explicit_project_root.resolve()

    run_dir = run_dir.resolve()
    parent = run_dir.parent
    if parent.name == "runs":
        return parent.parent.resolve()
    return Path.cwd().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not load as a dict: {path}")
    return data


def _import_builder(builder_spec: str) -> Callable[[dict[str, Any]], torch.nn.Module]:
    if ":" not in builder_spec:
        raise ValueError("--builder must be in the form module.submodule:function")
    module_name, fn_name = builder_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Builder is not callable: {builder_spec}")
    return fn


def _resolve_from_project_root(project_root: Path, raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _load_artifacts(run_dir: Path, project_root: Path) -> RunArtifacts:
    config_path = run_dir / "resolved_config.yaml"
    model_path = run_dir / "final_model.pt"
    preprocessor_path = run_dir / "tabular_preprocessor.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing tabular preprocessor: {preprocessor_path}")

    config = _load_yaml(config_path)

    target_column = config["schema"]["target_column"]
    id_column = config["schema"].get("id_column", "item_id")
    text_columns = list(config["fusion"]["text_columns"])
    feature_columns = list(config["selection"]["selected_numeric_features"])

    tabular_cfg = config["tabular_input"]
    feature_dir = _resolve_from_project_root(project_root, tabular_cfg["feature_dir"])
    dev_csv_path = feature_dir / tabular_cfg.get("dev_file", "dev_features.csv")
    if not dev_csv_path.exists():
        raise FileNotFoundError(f"Missing dev feature file: {dev_csv_path}")

    return RunArtifacts(
        run_dir=run_dir,
        config=config,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        dev_csv_path=dev_csv_path,
        target_column=target_column,
        id_column=id_column,
        text_columns=text_columns,
        feature_columns=feature_columns,
    )


def _load_preprocessor(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _apply_tabular_preprocessor(
    preprocessor: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    X_df = df[feature_columns].copy()

    if hasattr(preprocessor, "transform"):
        X = preprocessor.transform(X_df)
    else:
        raise TypeError("Loaded tabular preprocessor has no .transform() method")

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    transformed_feature_names: list[str]
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            transformed_feature_names = list(map(str, preprocessor.get_feature_names_out()))
        except Exception:
            transformed_feature_names = list(feature_columns)
    else:
        transformed_feature_names = list(feature_columns)

    if X.shape[1] != len(transformed_feature_names):
        transformed_feature_names = [f"f_{i}" for i in range(X.shape[1])]

    return X, transformed_feature_names


def _build_model(builder: Callable[[dict[str, Any]], torch.nn.Module], config: dict[str, Any]) -> torch.nn.Module:
    model = builder(config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Builder did not return a torch.nn.Module")
    return model


def _load_model_weights(
    model: torch.nn.Module,
    model_path: Path,
    device: torch.device,
    *,
    strict: bool,
) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Could not identify state dict in checkpoint. Expected raw state_dict, "
                "or keys like 'state_dict' / 'model_state_dict'."
            )
    else:
        raise TypeError("Unsupported checkpoint format for final_model.pt")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[warn] Missing state_dict keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] Unexpected state_dict keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.to(device)
    model.eval()
    return model


def _load_tokenizer(encoder_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(encoder_name)


def _tokenize_text_rows(
    df: pd.DataFrame,
    *,
    text_columns: list[str],
    encoder_name: str,
    max_length: int,
    join_mode: str,
    add_special_tokens: bool,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    tokenizer = _load_tokenizer(encoder_name)

    def build_text(row: pd.Series) -> str:
        parts = []
        for c in text_columns:
            val = row.get(c, "")
            if pd.isna(val):
                val = ""
            parts.append(str(val))
        if join_mode == "sep":
            sep_token = tokenizer.sep_token or " [SEP] "
            return f" {sep_token} ".join(parts)
        return " ".join(parts)

    texts = [build_text(row) for _, row in df[text_columns].iterrows()]

    encodings: list[dict[str, list[int]]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        encodings.append({k: v for k, v in enc.items()})

    merged: dict[str, torch.Tensor] = {}
    for key in encodings[0].keys():
        merged[key] = torch.cat([e[key] for e in encodings], dim=0)
    return merged


def _run_model_forward(
    model: torch.nn.Module,
    *,
    fixed_inputs: dict[str, torch.Tensor],
    tabular_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_inputs = {k: v.to(device) for k, v in fixed_inputs.items()}
    tabular_features = tabular_features.to(device)

    with torch.set_grad_enabled(True):
        try:
            output = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs.get("attention_mask"),
                tabular_features=tabular_features,
            )
        except TypeError:
            payload = dict(batch_inputs)
            payload["tabular_features"] = tabular_features
            output = model(payload)

        if isinstance(output, dict):
            for key in ("preds", "prediction", "logits", "output"):
                if key in output:
                    output = output[key]
                    break

        if not torch.is_tensor(output):
            raise TypeError(
                "Model output is not a tensor. Edit _run_model_forward() to match your project output format."
            )

        if output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        elif output.ndim != 1:
            raise ValueError(
                f"Expected scalar regression output of shape [batch] or [batch,1], got {tuple(output.shape)}"
            )

        return output


def _sample_indices(n_total: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    n_take = min(n_total, max(1, n_take))
    return np.sort(rng.choice(n_total, size=n_take, replace=False))


def _ensure_2d_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _make_beeswarm_plot(explanation: shap.Explanation, output_path: Path, max_display: int) -> None:
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _make_bar_plot(explanation: shap.Explanation, output_path: Path, max_display: int) -> None:
    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _make_scatter_plot(
    explanation: shap.Explanation,
    feature_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    shap.plots.scatter(explanation[:, feature_name], color=explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _save_local_explanations(
    explanation: shap.Explanation,
    item_ids: list[str],
    output_path: Path,
) -> None:
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    base_values = np.asarray(explanation.base_values)
    feature_names = list(map(str, explanation.feature_names))

    records: list[dict[str, Any]] = []
    for row_idx, item_id in enumerate(item_ids):
        row_values = values[row_idx]
        row_data = data[row_idx]
        order = np.argsort(np.abs(row_values))[::-1]
        contributions = [
            {
                "feature_name": feature_names[i],
                "feature_value": None if pd.isna(row_data[i]) else float(row_data[i]),
                "shap_value": float(row_values[i]),
                "abs_shap_value": float(abs(row_values[i])),
            }
            for i in order
        ]
        records.append(
            {
                "item_id": item_id,
                "base_value": float(base_values[row_idx]) if np.ndim(base_values) > 0 else float(base_values),
                "pred_minus_base_sum": float(np.sum(row_values)),
                "top_contributions": contributions[:20],
            }
        )

    _save_json(output_path, {"local_explanations": records})


def main() -> None:
    args = _parse_args()
    _set_random_seed(args.random_seed)

    run_dir = args.run_dir.resolve()
    project_root = _resolve_project_root(run_dir, args.project_root)
    output_dir = (args.output_dir or (run_dir / "shap_tabular")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.random_seed)

    artifacts = _load_artifacts(run_dir, project_root)
    builder = _import_builder(args.builder)

    print(f"[info] Loading dev features from: {artifacts.dev_csv_path}")
    dev_df = pd.read_csv(artifacts.dev_csv_path)

    required_columns = set(artifacts.text_columns + artifacts.feature_columns + [artifacts.id_column])
    missing_columns = [c for c in required_columns if c not in dev_df.columns]
    if missing_columns:
        raise KeyError(f"Dev dataframe is missing required columns: {missing_columns}")

    preprocessor = _load_preprocessor(artifacts.preprocessor_path)
    X_all, transformed_feature_names = _apply_tabular_preprocessor(
        preprocessor,
        dev_df,
        artifacts.feature_columns,
    )

    explain_idx = _sample_indices(len(dev_df), args.explain_size, rng)
    background_idx = _sample_indices(len(dev_df), args.background_size, rng)

    explain_df = dev_df.iloc[explain_idx].reset_index(drop=True)
    background_df = dev_df.iloc[background_idx].reset_index(drop=True)

    X_explain, transformed_feature_names_explain = _apply_tabular_preprocessor(
        preprocessor,
        explain_df,
        artifacts.feature_columns,
    )
    X_background, transformed_feature_names_bg = _apply_tabular_preprocessor(
        preprocessor,
        background_df,
        artifacts.feature_columns,
    )

    if transformed_feature_names_explain != transformed_feature_names_bg:
        raise ValueError("Transformed feature names differ between explain and background sets.")

    transformed_feature_names = transformed_feature_names_explain

    encoder_name = artifacts.config["fusion"]["encoder_name"]
    max_length = int(artifacts.config["fusion"].get("max_length", 128))
    join_mode = str(artifacts.config["fusion"].get("text_join_mode", "sep"))
    add_special_tokens = bool(artifacts.config["fusion"].get("add_special_tokens", True))

    print(f"[info] Rebuilding model via: {args.builder}")
    model = _build_model(builder, artifacts.config)
    model = _load_model_weights(
        model,
        artifacts.model_path,
        device,
        strict=args.strict_state_dict,
    )

    print("[info] Tokenizing background texts...")
    background_tokens = _tokenize_text_rows(
        background_df,
        text_columns=artifacts.text_columns,
        encoder_name=encoder_name,
        max_length=max_length,
        join_mode=join_mode,
        add_special_tokens=add_special_tokens,
        batch_size=args.batch_size,
    )

    print("[info] Tokenizing explain texts...")
    explain_tokens = _tokenize_text_rows(
        explain_df,
        text_columns=artifacts.text_columns,
        encoder_name=encoder_name,
        max_length=max_length,
        join_mode=join_mode,
        add_special_tokens=add_special_tokens,
        batch_size=args.batch_size,
    )

    background_tabular = torch.tensor(X_background, dtype=torch.float32, device=device)
    explain_tabular = torch.tensor(X_explain, dtype=torch.float32, device=device)

    # GradientExplainer explains one model/input pair at a time. Because the text
    # branch is fixed for a given dataset subset here, we explain the tabular input
    # while keeping the corresponding tokenized text fixed inside the wrapper.
    #
    # We create two wrappers so the background and explained batches use their own
    # aligned fixed text tensors.
    background_wrapper = TabularOnlyWrapper(
        model,
        fixed_inputs=background_tokens,
        device=device,
    ).to(device)
    explain_wrapper = TabularOnlyWrapper(
        model,
        fixed_inputs=explain_tokens,
        device=device,
    ).to(device)

    # We use the background wrapper only to establish the background distribution
    # for the tabular tensor. Then we run the same architecture with the explain
    # wrapper to get explanations for the explain set.
    #
    # Because SHAP explainers are tied to the provided model object, we instantiate
    # the explainer on the explain wrapper but use tabular background samples from
    # the same feature space. This works because the architecture and weights are
    # identical; only the fixed text tensors differ across wrappers.
    print("[info] Creating GradientExplainer...")
    explainer = shap.GradientExplainer(explain_wrapper, background_tabular)

    print("[info] Computing SHAP values...")
    shap_values = explainer.shap_values(explain_tabular)

    shap_values_arr = _ensure_2d_array(shap_values)
    base_values = explain_wrapper(explain_tabular).detach().cpu().numpy() - shap_values_arr.sum(axis=1)

    explanation = shap.Explanation(
        values=shap_values_arr,
        base_values=base_values,
        data=X_explain,
        feature_names=transformed_feature_names,
    )

    mean_abs = np.abs(shap_values_arr).mean(axis=0)
    shap_importance_df = pd.DataFrame(
        {
            "feature_name": transformed_feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values(["mean_abs_shap", "feature_name"], ascending=[False, True]).reset_index(drop=True)

    shap_importance_df.to_csv(output_dir / "tabular_shap_importance.csv", index=False)

    _make_beeswarm_plot(explanation, output_dir / "tabular_shap_beeswarm.png", args.max_display)
    _make_bar_plot(explanation, output_dir / "tabular_shap_bar.png", args.max_display)

    top_features = shap_importance_df["feature_name"].head(args.dependence_top_k).tolist()
    for feature_name in top_features:
        safe_name = feature_name.replace("/", "_")
        _make_scatter_plot(
            explanation,
            feature_name,
            output_dir / f"tabular_shap_scatter_{safe_name}.png",
        )

    local_item_ids = [x.strip() for x in args.local_item_ids.split(",") if x.strip()]
    if local_item_ids:
        local_mask = explain_df[artifacts.id_column].astype(str).isin(local_item_ids)
        if local_mask.any():
            local_rows = np.where(local_mask.to_numpy())[0]
            local_explanation = explanation[local_rows]
            local_ids = explain_df.iloc[local_rows][artifacts.id_column].astype(str).tolist()
            _save_local_explanations(
                local_explanation,
                local_ids,
                output_dir / "tabular_shap_local_explanations.json",
            )
        else:
            print("[warn] None of the requested --local-item-ids were present in the explained subset.")

    metadata = {
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "dev_csv_path": str(artifacts.dev_csv_path),
        "model_path": str(artifacts.model_path),
        "preprocessor_path": str(artifacts.preprocessor_path),
        "encoder_name": encoder_name,
        "n_dev_rows_total": int(len(dev_df)),
        "n_background_rows": int(len(background_df)),
        "n_explained_rows": int(len(explain_df)),
        "n_transformed_features": int(len(transformed_feature_names)),
        "top_global_feature": None if shap_importance_df.empty else str(shap_importance_df.iloc[0]["feature_name"]),
        "builder": args.builder,
        "device": str(device),
    }
    _save_json(output_dir / "tabular_shap_metadata.json", metadata)

    print(f"[done] Saved SHAP outputs to: {output_dir}")


if __name__ == "__main__":
    main()
