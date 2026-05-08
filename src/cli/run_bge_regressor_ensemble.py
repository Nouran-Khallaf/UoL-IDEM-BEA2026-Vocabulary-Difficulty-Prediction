from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class InputConfig:
    train_file: str
    dev_file: str | None = None


@dataclass(slots=True)
class SchemaConfig:
    target_column: str
    id_column: str
    l1_column: str | None = None


@dataclass(slots=True)
class TextConfig:
    text_columns: list[str]
    text_join_mode: str = "sep"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    normalize_embeddings: bool = False


@dataclass(slots=True)
class FeatureConfig:
    include_tabular: bool = False
    exclude_columns: list[str] | None = None
    use_pca: bool = False
    pca_dim: int = 256


@dataclass(slots=True)
class CVConfig:
    folds: int = 5
    shuffle: bool = True
    random_state: int = 42


@dataclass(slots=True)
class ModelConfig:
    regressors: list[str]
    create_mean_ensemble: bool = True
    create_stacking_ensemble: bool = True


@dataclass(slots=True)
class OutputConfig:
    output_dir: str


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    random_seed: int
    inputs: InputConfig
    schema: SchemaConfig
    text: TextConfig
    embeddings: EmbeddingConfig
    features: FeatureConfig
    cv: CVConfig
    model: ModelConfig
    outputs: OutputConfig


# ============================================================
# Config loading
# ============================================================

def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        random_seed=raw.get("random_seed", 42),
        inputs=InputConfig(**raw["inputs"]),
        schema=SchemaConfig(**raw["schema"]),
        text=TextConfig(**raw["text"]),
        embeddings=EmbeddingConfig(**raw["embeddings"]),
        features=FeatureConfig(**raw["features"]),
        cv=CVConfig(**raw["cv"]),
        model=ModelConfig(**raw["model"]),
        outputs=OutputConfig(**raw["outputs"]),
    )


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "kendall_tau": float(kendalltau(y_true, y_pred)[0]),
    }


def build_texts(
    df: pd.DataFrame,
    text_columns: list[str],
    text_join_mode: str,
) -> list[str]:
    sep = " [SEP] " if text_join_mode == "sep" else " "
    texts: list[str] = []

    for _, row in df.iterrows():
        parts: list[str] = []
        for col in text_columns:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            parts.append(str(value).strip())
        texts.append(sep.join(parts))

    return texts


def get_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    schema: SchemaConfig,
    text_columns: list[str],
    exclude_columns: list[str] | None,
) -> list[str]:
    excluded = set(exclude_columns or [])
    excluded.add(schema.target_column)
    excluded.add(schema.id_column)
    if schema.l1_column:
        excluded.add(schema.l1_column)

    for col in text_columns:
        excluded.add(col)

    numeric_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(numeric_cols)


def fit_transform_tabular(
    X_train_tab: np.ndarray,
    X_valid_tab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()

    X_train_tab = imputer.fit_transform(X_train_tab)
    X_valid_tab = imputer.transform(X_valid_tab)

    X_train_tab = scaler.fit_transform(X_train_tab)
    X_valid_tab = scaler.transform(X_valid_tab)

    return X_train_tab, X_valid_tab


def maybe_apply_pca(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    use_pca: bool,
    pca_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_pca:
        return X_train, X_valid

    n_components = min(pca_dim, X_train.shape[0], X_train.shape[1])
    if n_components < 2:
        return X_train, X_valid

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    return X_train_pca, X_valid_pca


# ============================================================
# Models
# ============================================================

def build_regressor(name: str, seed: int) -> Any:
    resolved = name.strip().lower()

    if resolved == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    if resolved == "gbr":
        return GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        )

    if resolved == "xgb":
        return XGBRegressor(
            random_state=seed,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
        )

    if resolved == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=5.0, epsilon=0.05, kernel="rbf")),
        ])

    raise ValueError(f"Unsupported regressor '{name}'.")


# ============================================================
# Main
# ============================================================

def run_experiment(cfg: ExperimentConfig) -> None:
    out_dir = ensure_dir(cfg.outputs.output_dir)

    train_df = pd.read_csv(cfg.inputs.train_file)
    y = train_df[cfg.schema.target_column].to_numpy(dtype=float)

    texts = build_texts(
        train_df,
        text_columns=cfg.text.text_columns,
        text_join_mode=cfg.text.text_join_mode,
    )

    print(f"Loaded train: {train_df.shape}")
    print(f"Encoding with: {cfg.embeddings.model_name}")

    encoder = SentenceTransformer(cfg.embeddings.model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=cfg.embeddings.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.embeddings.normalize_embeddings,
    ).astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")

    feature_columns: list[str] = []
    tabular_matrix: np.ndarray | None = None

    if cfg.features.include_tabular:
        feature_columns = get_numeric_feature_columns(
            train_df,
            schema=cfg.schema,
            text_columns=cfg.text.text_columns,
            exclude_columns=cfg.features.exclude_columns,
        )
        print(f"Using {len(feature_columns)} numeric tabular features.")
        tabular_matrix = train_df[feature_columns].to_numpy(dtype=np.float32)
    else:
        print("Embeddings-only mode.")

    kf = KFold(
        n_splits=cfg.cv.folds,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )

    regressor_names = [name.lower() for name in cfg.model.regressors]
    oof_preds = {
        name: np.zeros(len(train_df), dtype=np.float64)
        for name in regressor_names
    }

    fold_records: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(embeddings), start=1):
        print(f"\n========== Fold {fold}/{cfg.cv.folds} ==========")

        X_train_emb = embeddings[train_idx]
        X_valid_emb = embeddings[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        X_train_final = X_train_emb
        X_valid_final = X_valid_emb

        if tabular_matrix is not None:
            X_train_tab = tabular_matrix[train_idx]
            X_valid_tab = tabular_matrix[valid_idx]
            X_train_tab, X_valid_tab = fit_transform_tabular(X_train_tab, X_valid_tab)

            X_train_final = np.concatenate([X_train_emb, X_train_tab], axis=1)
            X_valid_final = np.concatenate([X_valid_emb, X_valid_tab], axis=1)

        X_train_final, X_valid_final = maybe_apply_pca(
            X_train_final,
            X_valid_final,
            use_pca=cfg.features.use_pca,
            pca_dim=cfg.features.pca_dim,
            seed=cfg.random_seed,
        )

        fold_result: dict[str, Any] = {"fold": fold, "metrics": {}}

        for name in regressor_names:
            model = build_regressor(name, cfg.random_seed)
            model.fit(X_train_final, y_train)
            preds = model.predict(X_valid_final)

            oof_preds[name][valid_idx] = preds
            metrics = compute_metrics(y_valid, preds)
            fold_result["metrics"][name] = metrics
            print(f"{name}: {metrics}")

        fold_records.append(fold_result)

    summary: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "feature_columns": feature_columns,
        "folds": fold_records,
        "oof_metrics": {},
    }

    for name in regressor_names:
        summary["oof_metrics"][name] = compute_metrics(y, oof_preds[name])

    pred_df = pd.DataFrame({
        cfg.schema.id_column: train_df[cfg.schema.id_column],
        cfg.schema.target_column: y,
    })

    for name in regressor_names:
        pred_df[f"{name}_pred_oof"] = oof_preds[name]

    if cfg.model.create_mean_ensemble and len(regressor_names) > 1:
        ensemble_oof = np.mean(
            np.column_stack([oof_preds[name] for name in regressor_names]),
            axis=1,
        )
        pred_df["mean_ensemble_pred_oof"] = ensemble_oof
        summary["oof_metrics"]["mean_ensemble"] = compute_metrics(y, ensemble_oof)

    if cfg.model.create_stacking_ensemble and len(regressor_names) > 1:
        base_oof = np.column_stack([oof_preds[name] for name in regressor_names])
        meta_model = Ridge(alpha=1.0)
        stacking_oof = np.zeros(len(train_df), dtype=np.float64)

        for train_idx, valid_idx in kf.split(base_oof):
            X_meta_train = base_oof[train_idx]
            X_meta_valid = base_oof[valid_idx]
            y_meta_train = y[train_idx]

            meta_model_fold = clone(meta_model)
            meta_model_fold.fit(X_meta_train, y_meta_train)
            stacking_oof[valid_idx] = meta_model_fold.predict(X_meta_valid)

        pred_df["stacking_ridge_pred_oof"] = stacking_oof
        summary["oof_metrics"]["stacking_ridge"] = compute_metrics(y, stacking_oof)

    pred_df.to_csv(out_dir / "train_oof_predictions.csv", index=False)
    save_json(summary, out_dir / "oof_metrics.json")

    print("\n========== OOF Summary ==========")
    for name, metrics in summary["oof_metrics"].items():
        print(name, metrics)

    if cfg.inputs.dev_file:
        dev_df = pd.read_csv(cfg.inputs.dev_file)
        dev_texts = build_texts(
            dev_df,
            text_columns=cfg.text.text_columns,
            text_join_mode=cfg.text.text_join_mode,
        )

        dev_embeddings = encoder.encode(
            dev_texts,
            batch_size=cfg.embeddings.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=cfg.embeddings.normalize_embeddings,
        ).astype(np.float32)

        dev_fold_preds = {name: [] for name in regressor_names}
        dev_meta_fold_preds: list[np.ndarray] = []

        for train_idx, _ in kf.split(embeddings):
            X_train_emb = embeddings[train_idx]
            y_train = y[train_idx]

            X_train_final = X_train_emb
            X_dev_final = dev_embeddings

            if tabular_matrix is not None:
                if not feature_columns:
                    raise ValueError("feature_columns is empty in tabular mode.")

                missing_cols = [c for c in feature_columns if c not in dev_df.columns]
                if missing_cols:
                    raise ValueError(f"Dev file missing feature columns: {missing_cols}")

                X_train_tab = tabular_matrix[train_idx]
                X_dev_tab = dev_df[feature_columns].to_numpy(dtype=np.float32)
                X_train_tab, X_dev_tab = fit_transform_tabular(X_train_tab, X_dev_tab)

                X_train_final = np.concatenate([X_train_emb, X_train_tab], axis=1)
                X_dev_final = np.concatenate([dev_embeddings, X_dev_tab], axis=1)

            X_train_final, X_dev_final = maybe_apply_pca(
                X_train_final,
                X_dev_final,
                use_pca=cfg.features.use_pca,
                pca_dim=cfg.features.pca_dim,
                seed=cfg.random_seed,
            )

            dev_base_preds_this_fold: list[np.ndarray] = []

            for name in regressor_names:
                model = build_regressor(name, cfg.random_seed)
                model.fit(X_train_final, y_train)
                preds = model.predict(X_dev_final)
                dev_fold_preds[name].append(preds)
                dev_base_preds_this_fold.append(preds)

            if cfg.model.create_stacking_ensemble and len(regressor_names) > 1:
                base_oof = np.column_stack([oof_preds[name] for name in regressor_names])
                meta_model = Ridge(alpha=1.0)
                meta_model.fit(base_oof[train_idx], y_train)

                dev_base_matrix = np.column_stack(dev_base_preds_this_fold)
                dev_meta_fold_preds.append(meta_model.predict(dev_base_matrix))

        dev_out = pd.DataFrame({
            cfg.schema.id_column: dev_df[cfg.schema.id_column],
        })

        for name in regressor_names:
            dev_out[f"{name}_pred"] = np.mean(
                np.column_stack(dev_fold_preds[name]),
                axis=1,
            )

        if cfg.model.create_mean_ensemble and len(regressor_names) > 1:
            dev_out["mean_ensemble_pred"] = np.mean(
                np.column_stack([dev_out[f"{name}_pred"] for name in regressor_names]),
                axis=1,
            )

        if cfg.model.create_stacking_ensemble and len(regressor_names) > 1:
            dev_out["stacking_ridge_pred"] = np.mean(
                np.column_stack(dev_meta_fold_preds),
                axis=1,
            )

        dev_out.to_csv(out_dir / "dev_predictions.csv", index=False)

        if cfg.schema.target_column in dev_df.columns:
            dev_y = dev_df[cfg.schema.target_column].to_numpy(dtype=float)
            dev_metrics: dict[str, Any] = {}

            for name in regressor_names:
                dev_metrics[name] = compute_metrics(
                    dev_y,
                    dev_out[f"{name}_pred"].to_numpy(),
                )

            if "mean_ensemble_pred" in dev_out.columns:
                dev_metrics["mean_ensemble"] = compute_metrics(
                    dev_y,
                    dev_out["mean_ensemble_pred"].to_numpy(),
                )

            if "stacking_ridge_pred" in dev_out.columns:
                dev_metrics["stacking_ridge"] = compute_metrics(
                    dev_y,
                    dev_out["stacking_ridge_pred"].to_numpy(),
                )

            save_json(dev_metrics, out_dir / "dev_metrics.json")

            print("\n========== Dev Summary ==========")
            for name, metrics in dev_metrics.items():
                print(name, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BGE embedding regressor experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()