from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


FEATURE_GROUPS: dict[str, set[str]] = {
    "frequency": {
        "kelly_rank",
        "kelly_rank_percentile",
        "kelly_points",
        "kelly_cefr",
        "kelly_cefr_num",
        "subtlex_wf",
        "subtlex_lg10wf",
        "subtlex_cd",
        "subtlex_lg10cd",
        "subtlex_zipf",
        "wf_value",
        "wf_zipf",
        "wf_cost",
        "wf_percentile",
        "wf_found",
    },
    "lexical_surface": {
        "target_len",
        "source_len",
        "clue_len",
        "clue_ratio",
        "pos_encoded",
        "syllables",
        "target_vowel_ratio",
        "target_consonant_ratio",
        "shared_char_ratio",
        "baseline_clue_overlap",
        "clue_hidden_chars",
        "en_target_lemma",
    },
    "retrieval_baseline": {
        "retrieval_target_prior",
        "retrieval_target_in_context",
        "retrieval_target_in_clue",
        "retrieval_candidate_count",
        "baseline_pred_len",
        "baseline_pred_matches_target",
    },
    "mlm_surprisal": {
        "mlm_log_prob",
        "mlm_rank",
        "mlm_entropy",
        "surprisal_masked",
        "surprisal_pll",
        "surprisal_subword_mean",
        "surprisal_subword_sum",
        "target_log_prob",
        "is_top_1",
        "target_rank",
        "prediction_entropy",
    },
    "cognate": {
        "weighted_levenshtein_sim",
        "cognate_sim",
        "L1_source_word_has_alternative",
        "L1_source_word_has_excluded_word",
        "cognet_source_alternative_english_count",
    },
    "semantic": {
        "semantic_usas_entropy_unweighted",
        "semantic_usas_entropy_weighted",
        "semantic_usas_domain_match",
        "semantic_semantic_shift",
    },
}


def infer_feature_group(feature_name: str) -> str:
    for group_name, members in FEATURE_GROUPS.items():
        if feature_name in members:
            return group_name

    lower = feature_name.lower()

    if any(k in lower for k in ["kelly", "subtlex", "wf_", "zipf", "freq", "frequency"]):
        return "frequency"
    if any(k in lower for k in ["len", "syll", "vowel", "consonant", "char", "lemma", "pos", "clue"]):
        return "lexical_surface"
    if "retrieval" in lower or "baseline_" in lower:
        return "retrieval_baseline"
    if any(k in lower for k in ["mlm", "surprisal", "log_prob", "entropy", "rank", "top_1"]):
        return "mlm_surprisal"
    if any(k in lower for k in ["cognate", "levenshtein", "cognet", "alternative", "excluded_word"]):
        return "cognate"
    if any(k in lower for k in ["semantic", "usas", "domain_match", "shift"]):
        return "semantic"

    return "other"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute feature-target correlations, grouped rankings, and SHAP analyses for train/dev processed feature files."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory containing train/dev feature CSVs and diagnostics files.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train_features.csv",
        help="Training feature file name.",
    )
    parser.add_argument(
        "--dev-file",
        type=str,
        default="dev_features.csv",
        help="Dev feature file name.",
    )
    parser.add_argument(
        "--train-diagnostics-file",
        type=str,
        default="train_feature_diagnostics.json",
        help="Training diagnostics JSON file name.",
    )
    parser.add_argument(
        "--dev-diagnostics-file",
        type=str,
        default="dev_feature_diagnostics.json",
        help="Dev diagnostics JSON file name.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="GLMM_score",
        help="Target column name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top N overall features to use in the main plot.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="abs_kendall_tau",
        choices=["abs_pearson_r", "abs_spearman_rho", "abs_kendall_tau"],
        help="Score column used for ranking and plotting.",
    )
    parser.add_argument(
        "--encode-kelly-cefr",
        action="store_true",
        help="Encode kelly_cefr into numeric kelly_cefr_num.",
    )
    parser.add_argument(
        "--include-all-dataframe-columns",
        action="store_true",
        help="Use all dataframe columns except target/id instead of diagnostics feature_columns.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=400,
        help="Number of trees for RandomForestRegressor used in SHAP analysis.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Optional max depth for RandomForestRegressor.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=1000,
        help="Maximum number of rows sampled for SHAP plotting.",
    )
    return parser.parse_args()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_corr_pair(
    x: pd.Series,
    y: pd.Series,
) -> tuple[int, float | None, float | None, float | None, float | None, float | None, float | None]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    n_valid = len(pair)

    if n_valid < 2:
        return n_valid, None, None, None, None, None, None

    x_vals = pair["x"].to_numpy(dtype=float)
    y_vals = pair["y"].to_numpy(dtype=float)

    if len(np.unique(x_vals)) < 2 or len(np.unique(y_vals)) < 2:
        return n_valid, None, None, None, None, None, None

    try:
        pearson_r, pearson_p = pearsonr(x_vals, y_vals)
    except Exception:
        pearson_r, pearson_p = None, None

    try:
        spearman_rho, spearman_p = spearmanr(x_vals, y_vals)
    except Exception:
        spearman_rho, spearman_p = None, None

    try:
        kendall_tau, kendall_p = kendalltau(x_vals, y_vals)
    except Exception:
        kendall_tau, kendall_p = None, None

    return n_valid, pearson_r, pearson_p, spearman_rho, spearman_p, kendall_tau, kendall_p


def _load_feature_columns(diag_path: Path) -> list[str]:
    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    feature_columns = diagnostics.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"No feature_columns found in diagnostics file: {diag_path}")
    return [str(c).strip() for c in feature_columns if str(c).strip()]


def _ensure_extra_features(
    feature_columns: list[str],
    df: pd.DataFrame,
    extra_features: list[str],
) -> list[str]:
    feature_columns = list(feature_columns)
    seen = set(feature_columns)

    for col in extra_features:
        if col in df.columns and col not in seen:
            feature_columns.append(col)
            seen.add(col)

    return feature_columns


def _maybe_add_encoded_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    encode_kelly_cefr: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df = df.copy()
    feature_columns = list(feature_columns)
    added_features: list[str] = []

    if encode_kelly_cefr and "kelly_cefr" in df.columns:
        cefr_map = {
            "UNK": 0,
            "A1": 1,
            "A2": 2,
            "B1": 3,
            "B2": 4,
            "C1": 5,
            "C2": 6,
        }
        df["kelly_cefr_num"] = df["kelly_cefr"].astype(str).str.upper().map(cefr_map)

        if "kelly_cefr_num" not in feature_columns:
            feature_columns.append("kelly_cefr_num")
            added_features.append("kelly_cefr_num")

    info = {
        "added_encoded_features": added_features,
        "encode_kelly_cefr": bool(encode_kelly_cefr),
    }
    return df, feature_columns, info


def _compute_correlation_df(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {dataset_name} dataframe.")

    y = _safe_numeric(df[target_column])
    rows: list[dict[str, Any]] = []

    skipped_missing: list[str] = []
    skipped_non_numeric_after_coercion: list[str] = []
    skipped_constant_or_insufficient: list[str] = []

    for feature_name in feature_columns:
        if feature_name == target_column:
            continue

        if feature_name not in df.columns:
            skipped_missing.append(feature_name)
            continue

        x = _safe_numeric(df[feature_name])

        if x.notna().sum() < 2:
            skipped_non_numeric_after_coercion.append(feature_name)
            continue

        n_valid, pearson_r, pearson_p, spearman_rho, spearman_p, kendall_tau, kendall_p = _safe_corr_pair(x, y)

        if pearson_r is None and spearman_rho is None and kendall_tau is None:
            skipped_constant_or_insufficient.append(feature_name)
            continue

        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": infer_feature_group(feature_name),
                "dataset_name": dataset_name,
                "target_column": target_column,
                "n_valid": int(n_valid),
                "pearson_r": np.nan if pearson_r is None else float(pearson_r),
                "pearson_p": np.nan if pearson_p is None else float(pearson_p),
                "abs_pearson_r": np.nan if pearson_r is None else float(abs(pearson_r)),
                "spearman_rho": np.nan if spearman_rho is None else float(spearman_rho),
                "spearman_p": np.nan if spearman_p is None else float(spearman_p),
                "abs_spearman_rho": np.nan if spearman_rho is None else float(abs(spearman_rho)),
                "kendall_tau": np.nan if kendall_tau is None else float(kendall_tau),
                "kendall_p": np.nan if kendall_p is None else float(kendall_p),
                "abs_kendall_tau": np.nan if kendall_tau is None else float(abs(kendall_tau)),
            }
        )

    corr_df = pd.DataFrame(rows)

    if corr_df.empty:
        return corr_df

    corr_df = corr_df.sort_values(
        ["abs_kendall_tau", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    corr_df["rank_by_abs_pearson"] = (
        corr_df["abs_pearson_r"].rank(method="min", ascending=False).astype("Int64")
    )
    corr_df["rank_by_abs_spearman"] = (
        corr_df["abs_spearman_rho"].rank(method="min", ascending=False).astype("Int64")
    )
    corr_df["rank_by_abs_kendall"] = (
        corr_df["abs_kendall_tau"].rank(method="min", ascending=False).astype("Int64")
    )

    return corr_df


def _prepare_top_n(df: pd.DataFrame, *, score_column: str, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in dataframe.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    plot_df = df.sort_values(score_column, ascending=False).head(top_n).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    return plot_df


def _plot_top_features(
    corr_df: pd.DataFrame,
    *,
    output_path: Path,
    top_n: int,
    score_column: str,
    title: str,
) -> None:
    if corr_df.empty:
        return

    plot_df = _prepare_top_n(corr_df, score_column=score_column, top_n=top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature_name"], plot_df[score_column])
    plt.xlabel(score_column)
    plt.ylabel("feature_name")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_group_rankings(
    corr_df: pd.DataFrame,
    *,
    output_dir: Path,
    prefix: str,
    score_column: str = "abs_kendall_tau",
) -> dict[str, Any]:
    if corr_df.empty:
        empty_df = pd.DataFrame(columns=["feature_group", "feature_name", score_column])
        empty_df.to_csv(output_dir / f"{prefix}_top5_features_per_group.csv", index=False)
        empty_df.to_csv(output_dir / f"{prefix}_top1_feature_per_group.csv", index=False)
        return {"top5_per_group": {}, "top1_per_group": {}}

    ranked = corr_df.sort_values([score_column, "feature_name"], ascending=[False, True]).copy()

    top5_df = ranked.groupby("feature_group", group_keys=False).head(5).reset_index(drop=True)
    top1_df = ranked.groupby("feature_group", group_keys=False).head(1).reset_index(drop=True)

    top5_df.to_csv(output_dir / f"{prefix}_top5_features_per_group.csv", index=False)
    top1_df.to_csv(output_dir / f"{prefix}_top1_feature_per_group.csv", index=False)

    summary = {
        "top5_per_group": {
            group_name: group_rows[["feature_name", score_column]].to_dict(orient="records")
            for group_name, group_rows in top5_df.groupby("feature_group")
        },
        "top1_per_group": {
            group_name: str(group_rows.iloc[0]["feature_name"])
            for group_name, group_rows in top1_df.groupby("feature_group")
        },
    }

    with (output_dir / f"{prefix}_group_feature_rankings.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _build_numeric_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    usable_cols: list[str] = []

    for col in feature_columns:
        if col == target_column or col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 2:
            continue
        if x.nunique(dropna=True) < 2:
            continue

        usable_cols.append(col)

    X = pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in usable_cols})
    y = pd.to_numeric(df[target_column], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y


def _fit_random_forest_for_shap(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    random_seed: int,
    n_estimators: int,
    max_depth: int | None,
) -> tuple[RandomForestRegressor, pd.DataFrame]:
    X, y = _build_numeric_feature_matrix(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_imp_df, y.to_numpy(dtype=float))

    return model, X_imp_df


def _save_shap_outputs(
    *,
    model: RandomForestRegressor,
    X_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    top1_per_group_features: list[str],
    max_samples: int,
) -> None:
    summary_path = output_dir / f"{prefix}_shap_summary.json"

    if not SHAP_AVAILABLE:
        info = {
            "shap_available": False,
            "message": "Install shap to generate SHAP outputs.",
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        return

    shap_df = X_df.copy()
    if len(shap_df) > max_samples:
        shap_df = shap_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_df)

    shap_importance_df = pd.DataFrame(
        {
            "feature_name": shap_df.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "feature_group": [infer_feature_group(c) for c in shap_df.columns],
        }
    ).sort_values(["mean_abs_shap", "feature_name"], ascending=[False, True]).reset_index(drop=True)

    shap_importance_df.to_csv(output_dir / f"{prefix}_shap_feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, shap_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_df, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    dependence_files: list[str] = []
    for feature_name in top1_per_group_features:
        if feature_name not in shap_df.columns:
            continue
        safe_name = feature_name.replace("/", "_")
        plt.figure()
        shap.dependence_plot(feature_name, shap_values, shap_df, show=False)
        plt.tight_layout()
        out_name = f"{prefix}_shap_dependence_{safe_name}.png"
        plt.savefig(output_dir / out_name, dpi=200, bbox_inches="tight")
        plt.close()
        dependence_files.append(out_name)

    summary = {
        "shap_available": True,
        "n_rows_used": int(len(shap_df)),
        "n_features_used": int(shap_df.shape[1]),
        "top_global_feature": None if shap_importance_df.empty else str(shap_importance_df.iloc[0]["feature_name"]),
        "dependence_plots": dependence_files,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _save_outputs(
    corr_df: pd.DataFrame,
    *,
    output_dir: Path,
    prefix: str,
    target_column: str,
    top_n: int,
    score_column: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(output_dir / f"{prefix}_feature_target_correlation.csv", index=False)

    summary = {
        "dataset_name": prefix,
        "target_column": target_column,
        "n_features_analyzed": int(len(corr_df)),
        "top_feature_by_abs_kendall": None if corr_df.empty else str(corr_df.iloc[0]["feature_name"]),
        "top_abs_kendall": None if corr_df.empty else float(corr_df.iloc[0]["abs_kendall_tau"]),
    }

    with (output_dir / f"{prefix}_feature_target_correlation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not corr_df.empty:
        _plot_top_features(
            corr_df,
            output_path=output_dir / f"{prefix}_feature_target_correlation_top{top_n}.png",
            top_n=top_n,
            score_column=score_column,
            title=f"{prefix} feature-target correlation",
        )


def main() -> None:
    args = _parse_args()
    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_features = [
        "L1_source_word_has_alternative",
        "L1_source_word_has_excluded_word",
        "cognet_source_alternative_english_count",
    ]

    train_df = pd.read_csv(feature_dir / args.train_file)

    if args.include_all_dataframe_columns:
        train_feature_columns = [
            c for c in train_df.columns
            if c not in {args.target_column, "item_id"}
        ]
    else:
        train_feature_columns = _load_feature_columns(feature_dir / args.train_diagnostics_file)

    train_feature_columns = _ensure_extra_features(
        train_feature_columns,
        train_df,
        extra_features,
    )

    train_df, train_feature_columns, _ = _maybe_add_encoded_features(
        train_df,
        train_feature_columns,
        encode_kelly_cefr=args.encode_kelly_cefr,
    )

    train_corr_df = _compute_correlation_df(
        train_df,
        feature_columns=train_feature_columns,
        target_column=args.target_column,
        dataset_name="train",
    )

    _save_outputs(
        train_corr_df,
        output_dir=output_dir,
        prefix="train",
        target_column=args.target_column,
        top_n=args.top_n,
        score_column=args.score_column,
    )

    train_group_summary = _save_group_rankings(
        train_corr_df,
        output_dir=output_dir,
        prefix="train",
        score_column="abs_kendall_tau",
    )

    train_model, train_X_df = _fit_random_forest_for_shap(
        train_df,
        feature_columns=train_feature_columns,
        target_column=args.target_column,
        random_seed=args.random_seed,
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
    )

    _save_shap_outputs(
        model=train_model,
        X_df=train_X_df,
        output_dir=output_dir,
        prefix="train",
        top1_per_group_features=list(train_group_summary["top1_per_group"].values()),
        max_samples=args.shap_max_samples,
    )

    dev_path = feature_dir / args.dev_file
    dev_diag_path = feature_dir / args.dev_diagnostics_file

    if dev_path.exists():
        dev_df = pd.read_csv(dev_path)

        if args.include_all_dataframe_columns:
            dev_feature_columns = [
                c for c in dev_df.columns
                if c not in {args.target_column, "item_id"}
            ]
        elif dev_diag_path.exists():
            dev_feature_columns = _load_feature_columns(dev_diag_path)
        else:
            dev_feature_columns = [
                c for c in dev_df.columns
                if c not in {args.target_column, "item_id"}
            ]

        dev_feature_columns = _ensure_extra_features(
            dev_feature_columns,
            dev_df,
            extra_features,
        )

        dev_df, dev_feature_columns, _ = _maybe_add_encoded_features(
            dev_df,
            dev_feature_columns,
            encode_kelly_cefr=args.encode_kelly_cefr,
        )

        dev_corr_df = _compute_correlation_df(
            dev_df,
            feature_columns=dev_feature_columns,
            target_column=args.target_column,
            dataset_name="dev",
        )

        _save_outputs(
            dev_corr_df,
            output_dir=output_dir,
            prefix="dev",
            target_column=args.target_column,
            top_n=args.top_n,
            score_column=args.score_column,
        )

        _save_group_rankings(
            dev_corr_df,
            output_dir=output_dir,
            prefix="dev",
            score_column="abs_kendall_tau",
        )

    print(f"Saved correlation, grouped rankings, and SHAP outputs to: {output_dir}")


if __name__ == "__main__":
    main()