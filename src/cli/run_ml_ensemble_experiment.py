from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================================================
# Metrics
# =========================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}

    try:
        result["pearson"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        result["pearson"] = float("nan")

    try:
        result["spearman"] = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        result["spearman"] = float("nan")

    try:
        result["kendall_tau"] = float(kendalltau(y_true, y_pred)[0])
    except Exception:
        result["kendall_tau"] = float("nan")

    result["rmse"] = rmse(y_true, y_pred)
    return result


# =========================================================
# Config / args
# =========================================================

@dataclass(slots=True)
class ExperimentConfig:
    feature_dir: Path
    output_dir: Path
    target_col: str
    id_col: str | None
    train_file: str
    dev_file: str
    n_splits: int
    seed: int
    use_xgb: bool
    feature_include_prefixes: list[str] | None
    feature_exclude_prefixes: list[str] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classical ML regressors on numeric engineered BEA features and build an ensemble."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-col", type=str, default="GLMM_score")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--train-file", type=str, default="train_features.csv")
    parser.add_argument("--dev-file", type=str, default="dev_features.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-xgb", action="store_true")
    parser.add_argument(
        "--include-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Only keep feature columns starting with any of these prefixes.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Drop feature columns starting with any of these prefixes.",
    )
    return parser.parse_args()


# =========================================================
# Data helpers
# =========================================================

NON_FEATURE_COLUMNS = {
    "GLMM_score",
    "label",
    "split",
    "fold",
    "id",
    "item_id",
    "ID",
}


def load_data(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = config.feature_dir / config.train_file
    dev_path = config.feature_dir / config.dev_file

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training features: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing dev features: {dev_path}")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    if config.target_col not in train_df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in train file.")
    if config.target_col not in dev_df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in dev file.")

    return train_df, dev_df


def select_feature_columns(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    config: ExperimentConfig,
) -> list[str]:
    shared_cols = [c for c in train_df.columns if c in dev_df.columns]

    feature_cols = [
        c for c in shared_cols
        if c not in NON_FEATURE_COLUMNS
        and c != config.target_col
        and c != config.id_col
    ]

    if config.feature_include_prefixes:
        feature_cols = [
            c for c in feature_cols
            if any(c.startswith(prefix) for prefix in config.feature_include_prefixes)
        ]

    if config.feature_exclude_prefixes:
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(prefix) for prefix in config.feature_exclude_prefixes)
        ]

    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]

    return sorted(feature_cols)


def build_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> ColumnTransformer:
    numeric_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[col])
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


# =========================================================
# Models
# =========================================================

def get_base_models(seed: int, use_xgb: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "ridge": Ridge(alpha=3.0),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000, random_state=seed),
        "svr": SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale",
        ),
        "hgb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=seed,
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if use_xgb and HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

    return models


def build_model_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# =========================================================
# Training / OOF / Dev
# =========================================================

def train_base_models(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    X_train = train_df[feature_cols]
    y_train = train_df[config.target_col].to_numpy(dtype=float)
    X_dev = dev_df[feature_cols]
    y_dev = dev_df[config.target_col].to_numpy(dtype=float)

    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    preprocessor = build_preprocessor(train_df, feature_cols)
    models = get_base_models(config.seed, config.use_xgb)

    all_results: dict[str, Any] = {}
    oof_pred_frame = pd.DataFrame(index=train_df.index)
    dev_pred_frame = pd.DataFrame(index=dev_df.index)

    for model_name, estimator in models.items():
        fold_oof = np.zeros(len(train_df), dtype=float)
        fold_dev_preds: list[np.ndarray] = []
        fold_metrics: list[dict[str, float]] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train[va_idx]

            pipeline = build_model_pipeline(preprocessor, clone(estimator))
            pipeline.fit(X_tr, y_tr)

            va_pred = pipeline.predict(X_va)
            fold_oof[va_idx] = va_pred

            dev_pred = pipeline.predict(X_dev)
            fold_dev_preds.append(dev_pred)

            metrics = safe_corr(y_va, va_pred)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        oof_metrics = safe_corr(y_train, fold_oof)
        mean_dev_pred = np.mean(np.vstack(fold_dev_preds), axis=0)
        dev_metrics = safe_corr(y_dev, mean_dev_pred)

        all_results[model_name] = {
            "fold_metrics": fold_metrics,
            "oof_metrics": oof_metrics,
            "dev_metrics": dev_metrics,
        }

        oof_pred_frame[f"{model_name}_oof"] = fold_oof
        dev_pred_frame[f"{model_name}_dev"] = mean_dev_pred

        print(f"\n=== {model_name} ===")
        print("OOF:", json.dumps(oof_metrics, indent=2))
        print("DEV:", json.dumps(dev_metrics, indent=2))

    return {
        "results": all_results,
        "oof_pred_frame": oof_pred_frame,
        "dev_pred_frame": dev_pred_frame,
    }


# =========================================================
# Ensembles
# =========================================================

def build_inverse_rmse_weights(results: dict[str, Any]) -> dict[str, float]:
    raw_weights: dict[str, float] = {}

    for model_name, payload in results.items():
        score = payload["oof_metrics"]["rmse"]
        raw_weights[model_name] = 1.0 / max(score, 1e-8)

    total = sum(raw_weights.values())
    return {k: float(v / total) for k, v in raw_weights.items()}


def weighted_average_predictions(
    pred_frame: pd.DataFrame,
    weights: dict[str, float],
    suffix: str,
) -> np.ndarray:
    preds = np.zeros(len(pred_frame), dtype=float)
    for model_name, weight in weights.items():
        preds += weight * pred_frame[f"{model_name}_{suffix}"].to_numpy(dtype=float)
    return preds


def fit_stacking_ensemble(
    oof_pred_frame: pd.DataFrame,
    y_train: np.ndarray,
    dev_pred_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    stacker = LinearRegression()
    stacker.fit(oof_pred_frame.to_numpy(dtype=float), y_train)

    oof_stack = stacker.predict(oof_pred_frame.to_numpy(dtype=float))
    dev_stack = stacker.predict(dev_pred_frame.to_numpy(dtype=float))

    coef_map = {
        col: float(coef)
        for col, coef in zip(oof_pred_frame.columns, stacker.coef_)
    }
    coef_map["intercept"] = float(stacker.intercept_)

    return oof_stack, dev_stack, coef_map


# =========================================================
# Save
# =========================================================

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        id_col=args.id_col,
        train_file=args.train_file,
        dev_file=args.dev_file,
        n_splits=args.n_splits,
        seed=args.seed,
        use_xgb=args.use_xgb,
        feature_include_prefixes=args.include_prefixes,
        feature_exclude_prefixes=args.exclude_prefixes,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df = load_data(config)
    feature_cols = select_feature_columns(train_df, dev_df, config)

    if not feature_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    print(f"Using {len(feature_cols)} numeric feature columns.")

    train_payload = train_base_models(
        train_df=train_df,
        dev_df=dev_df,
        feature_cols=feature_cols,
        config=config,
    )

    results = train_payload["results"]
    oof_pred_frame = train_payload["oof_pred_frame"]
    dev_pred_frame = train_payload["dev_pred_frame"]

    y_train = train_df[config.target_col].to_numpy(dtype=float)
    y_dev = dev_df[config.target_col].to_numpy(dtype=float)

    weights = build_inverse_rmse_weights(results)
    weighted_oof = weighted_average_predictions(oof_pred_frame, weights, suffix="oof")
    weighted_dev = weighted_average_predictions(dev_pred_frame, weights, suffix="dev")

    weighted_oof_metrics = safe_corr(y_train, weighted_oof)
    weighted_dev_metrics = safe_corr(y_dev, weighted_dev)

    stack_oof, stack_dev, stack_coefs = fit_stacking_ensemble(
        oof_pred_frame=oof_pred_frame,
        y_train=y_train,
        dev_pred_frame=dev_pred_frame,
    )
    stack_oof_metrics = safe_corr(y_train, stack_oof)
    stack_dev_metrics = safe_corr(y_dev, stack_dev)

    summary = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "base_models": results,
        "weighted_ensemble": {
            "weights": weights,
            "oof_metrics": weighted_oof_metrics,
            "dev_metrics": weighted_dev_metrics,
        },
        "stacking_ensemble": {
            "coefficients": stack_coefs,
            "oof_metrics": stack_oof_metrics,
            "dev_metrics": stack_dev_metrics,
        },
    }

    train_out = train_df.copy()
    dev_out = dev_df.copy()

    for col in oof_pred_frame.columns:
        train_out[col] = oof_pred_frame[col]

    for col in dev_pred_frame.columns:
        dev_out[col] = dev_pred_frame[col]

    train_out["ensemble_weighted_oof"] = weighted_oof
    dev_out["ensemble_weighted_dev"] = weighted_dev

    train_out["ensemble_stack_oof"] = stack_oof
    dev_out["ensemble_stack_dev"] = stack_dev

    train_out.to_csv(config.output_dir / "train_predictions_with_oof.csv", index=False)
    dev_out.to_csv(config.output_dir / "dev_predictions_with_ensemble.csv", index=False)
    oof_pred_frame.to_csv(config.output_dir / "oof_base_model_predictions.csv", index=False)
    dev_pred_frame.to_csv(config.output_dir / "dev_base_model_predictions.csv", index=False)
    save_json(config.output_dir / "metrics_summary.json", summary)

    print("\n=== Weighted Ensemble ===")
    print("OOF:", json.dumps(weighted_oof_metrics, indent=2))
    print("DEV:", json.dumps(weighted_dev_metrics, indent=2))

    print("\n=== Stacking Ensemble ===")
    print("OOF:", json.dumps(stack_oof_metrics, indent=2))
    print("DEV:", json.dumps(stack_dev_metrics, indent=2))

    best_name = "weighted_ensemble"
    best_rmse = weighted_dev_metrics["rmse"]
    if stack_dev_metrics["rmse"] < best_rmse:
        best_name = "stacking_ensemble"
        best_rmse = stack_dev_metrics["rmse"]

    print(f"\nBest dev ensemble: {best_name} (RMSE={best_rmse:.6f})")


if __name__ == "__main__":
    main()