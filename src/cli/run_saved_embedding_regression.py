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


@dataclass(slots=True)
class InputConfig:
    train_file: str
    dev_file: str | None = None
    train_embeddings_file: str = ""
    dev_embeddings_file: str | None = None


@dataclass(slots=True)
class SchemaConfig:
    target_column: str
    id_column: str
    l1_column: str | None = None


@dataclass(slots=True)
class FeatureConfig:
    mode: str = "embeddings_only"  # embeddings_only | embeddings_plus_tabular
    exclude_columns: list[str] | None = None
    pca_mode: str = "none"  # none | use_saved_pca | fit_within_fold
    saved_pca_dim: int | None = None
    fold_pca_dim: int = 256


@dataclass(slots=True)
class RegressorConfig:
    type: str = "ridge"  # ridge | gbr | xgb | svr | average_ensemble
    use_stacking: bool = False
    stacking_alpha: float = 1.0


@dataclass(slots=True)
class CVConfig:
    folds: int = 5
    shuffle: bool = True
    random_state: int = 42


@dataclass(slots=True)
class OutputConfig:
    output_dir: str


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    random_seed: int
    inputs: InputConfig
    schema: SchemaConfig
    features: FeatureConfig
    regressor: RegressorConfig
    cv: CVConfig
    outputs: OutputConfig


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        random_seed=raw.get("random_seed", 42),
        inputs=InputConfig(**raw["inputs"]),
        schema=SchemaConfig(**raw["schema"]),
        features=FeatureConfig(**raw["features"]),
        regressor=RegressorConfig(**raw["regressor"]),
        cv=CVConfig(**raw["cv"]),
        outputs=OutputConfig(**raw["outputs"]),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "kendall_tau": float(kendalltau(y_true, y_pred)[0]),
    }


def get_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    schema: SchemaConfig,
    exclude_columns: list[str] | None,
) -> list[str]:
    excluded = set(exclude_columns or [])
    excluded.add(schema.target_column)
    excluded.add(schema.id_column)
    if schema.l1_column:
        excluded.add(schema.l1_column)

    numeric_cols = [
        c for c in df.columns
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


def maybe_apply_fold_pca(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    pca_mode: str,
    fold_pca_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pca_mode != "fit_within_fold":
        return X_train, X_valid

    n_components = min(fold_pca_dim, X_train.shape[0], X_train.shape[1])
    if n_components < 2:
        return X_train, X_valid

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    return X_train_pca, X_valid_pca


def build_all_regressors(seed: int) -> dict[str, Any]:
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "gbr": GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
        ),
        "xgb": XGBRegressor(
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
        ),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=5.0, epsilon=0.05, kernel="rbf")),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regressors from saved embeddings.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg.outputs.output_dir)

    train_df = pd.read_csv(cfg.inputs.train_file)
    y = train_df[cfg.schema.target_column].to_numpy(dtype=float)

    train_embeddings = np.load(cfg.inputs.train_embeddings_file).astype(np.float32)
    dev_df = pd.read_csv(cfg.inputs.dev_file) if cfg.inputs.dev_file else None
    dev_embeddings = (
        np.load(cfg.inputs.dev_embeddings_file).astype(np.float32)
        if cfg.inputs.dev_embeddings_file
        else None
    )

    feature_columns: list[str] = []
    if cfg.features.mode == "embeddings_plus_tabular":
        feature_columns = get_numeric_feature_columns(
            train_df,
            schema=cfg.schema,
            exclude_columns=cfg.features.exclude_columns,
        )

    all_regressors = build_all_regressors(cfg.random_seed)
    selected = ["ridge", "gbr", "xgb", "svr"] if cfg.regressor.type == "average_ensemble" else [cfg.regressor.type]

    kf = KFold(
        n_splits=cfg.cv.folds,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )

    oof_preds = {name: np.zeros(len(train_df), dtype=np.float64) for name in selected}
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_embeddings), start=1):
        X_train_emb = train_embeddings[train_idx]
        X_valid_emb = train_embeddings[valid_idx]

        if cfg.features.mode == "embeddings_only":
            X_train = X_train_emb
            X_valid = X_valid_emb
        elif cfg.features.mode == "embeddings_plus_tabular":
            X_train_tab = train_df.iloc[train_idx][feature_columns].to_numpy(dtype=np.float32)
            X_valid_tab = train_df.iloc[valid_idx][feature_columns].to_numpy(dtype=np.float32)
            X_train_tab, X_valid_tab = fit_transform_tabular(X_train_tab, X_valid_tab)
            X_train = np.concatenate([X_train_emb, X_train_tab], axis=1)
            X_valid = np.concatenate([X_valid_emb, X_valid_tab], axis=1)
        else:
            raise ValueError(f"Unsupported feature mode: {cfg.features.mode}")

        X_train, X_valid = maybe_apply_fold_pca(
            X_train,
            X_valid,
            pca_mode=cfg.features.pca_mode,
            fold_pca_dim=cfg.features.fold_pca_dim,
        )

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        fold_record = {"fold": fold, "metrics": {}}
        for name in selected:
            model = clone(all_regressors[name])
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            oof_preds[name][valid_idx] = preds
            fold_record["metrics"][name] = compute_metrics(y_valid, preds)
        fold_metrics.append(fold_record)

    results: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "feature_mode": cfg.features.mode,
        "pca_mode": cfg.features.pca_mode,
        "feature_columns": feature_columns,
        "folds": fold_metrics,
        "oof_metrics": {},
    }

    pred_df = pd.DataFrame({
        cfg.schema.id_column: train_df[cfg.schema.id_column],
        cfg.schema.target_column: y,
    })

    for name in selected:
        pred_df[f"{name}_pred_oof"] = oof_preds[name]
        results["oof_metrics"][name] = compute_metrics(y, oof_preds[name])

    if cfg.regressor.type == "average_ensemble":
        ensemble_oof = np.mean(np.column_stack([oof_preds[name] for name in selected]), axis=1)
        pred_df["mean_ensemble_pred_oof"] = ensemble_oof
        results["oof_metrics"]["mean_ensemble"] = compute_metrics(y, ensemble_oof)

        if cfg.regressor.use_stacking:
            stack_X = np.column_stack([oof_preds[name] for name in selected])
            stacker = Ridge(alpha=cfg.regressor.stacking_alpha)
            stacker.fit(stack_X, y)
            stacked_oof = stacker.predict(stack_X)
            pred_df["stacked_ridge_pred_oof"] = stacked_oof
            results["oof_metrics"]["stacked_ridge"] = compute_metrics(y, stacked_oof)

    pred_df.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    save_json(results, output_dir / "oof_metrics.json")

    if dev_df is not None and dev_embeddings is not None:
        dev_fold_preds = {name: [] for name in selected}

        for train_idx, _ in kf.split(train_embeddings):
            X_train_emb = train_embeddings[train_idx]

            if cfg.features.mode == "embeddings_only":
                X_train = X_train_emb
                X_dev = dev_embeddings
            else:
                X_train_tab = train_df.iloc[train_idx][feature_columns].to_numpy(dtype=np.float32)
                X_dev_tab = dev_df[feature_columns].to_numpy(dtype=np.float32)
                X_train_tab, X_dev_tab = fit_transform_tabular(X_train_tab, X_dev_tab)
                X_train = np.concatenate([X_train_emb, X_train_tab], axis=1)
                X_dev = np.concatenate([dev_embeddings, X_dev_tab], axis=1)

            X_train, X_dev = maybe_apply_fold_pca(
                X_train,
                X_dev,
                pca_mode=cfg.features.pca_mode,
                fold_pca_dim=cfg.features.fold_pca_dim,
            )

            y_train = y[train_idx]

            for name in selected:
                model = clone(all_regressors[name])
                model.fit(X_train, y_train)
                preds = model.predict(X_dev)
                dev_fold_preds[name].append(preds)

        dev_pred_df = pd.DataFrame({
            cfg.schema.id_column: dev_df[cfg.schema.id_column],
        })

        for name in selected:
            dev_pred_df[f"{name}_pred"] = np.mean(np.column_stack(dev_fold_preds[name]), axis=1)

        if cfg.regressor.type == "average_ensemble":
            dev_pred_df["mean_ensemble_pred"] = np.mean(
                np.column_stack([dev_pred_df[f"{name}_pred"] for name in selected]),
                axis=1,
            )

            if cfg.regressor.use_stacking:
                stack_X = np.column_stack([oof_preds[name] for name in selected])
                stacker = Ridge(alpha=cfg.regressor.stacking_alpha)
                stacker.fit(stack_X, y)

                dev_stack_X = np.column_stack([dev_pred_df[f"{name}_pred"] for name in selected])
                dev_pred_df["stacked_ridge_pred"] = stacker.predict(dev_stack_X)

        if cfg.schema.target_column in dev_df.columns:
            dev_y = dev_df[cfg.schema.target_column].to_numpy(dtype=float)
            dev_metrics: dict[str, Any] = {}

            for name in selected:
                dev_metrics[name] = compute_metrics(dev_y, dev_pred_df[f"{name}_pred"].to_numpy())

            if cfg.regressor.type == "average_ensemble":
                dev_metrics["mean_ensemble"] = compute_metrics(dev_y, dev_pred_df["mean_ensemble_pred"].to_numpy())
                if cfg.regressor.use_stacking:
                    dev_metrics["stacked_ridge"] = compute_metrics(dev_y, dev_pred_df["stacked_ridge_pred"].to_numpy())

            save_json(dev_metrics, output_dir / "dev_metrics.json")

        dev_pred_df.to_csv(output_dir / "dev_predictions.csv", index=False)


if __name__ == "__main__":
    main()