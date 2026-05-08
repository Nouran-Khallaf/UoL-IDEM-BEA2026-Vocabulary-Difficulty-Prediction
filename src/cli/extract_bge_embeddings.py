from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


@dataclass(slots=True)
class InputConfig:
    train_file: str
    dev_file: str | None = None


@dataclass(slots=True)
class SchemaConfig:
    id_column: str
    target_column: str | None = None
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
class PCAConfig:
    save_global_pca: bool = True
    dimensions: list[int] | None = None


@dataclass(slots=True)
class OutputConfig:
    output_dir: str
    prefix: str = "bge"


@dataclass(slots=True)
class ExtractionConfig:
    experiment_name: str
    inputs: InputConfig
    schema: SchemaConfig
    text: TextConfig
    embeddings: EmbeddingConfig
    pca: PCAConfig
    outputs: OutputConfig


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> ExtractionConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExtractionConfig(
        experiment_name=raw["experiment_name"],
        inputs=InputConfig(**raw["inputs"]),
        schema=SchemaConfig(**raw["schema"]),
        text=TextConfig(**raw["text"]),
        embeddings=EmbeddingConfig(**raw["embeddings"]),
        pca=PCAConfig(**raw["pca"]),
        outputs=OutputConfig(**raw["outputs"]),
    )


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


def build_index_df(
    df: pd.DataFrame,
    *,
    schema: SchemaConfig,
    texts: list[str],
) -> pd.DataFrame:
    cols = [schema.id_column]
    if schema.l1_column and schema.l1_column in df.columns:
        cols.append(schema.l1_column)
    if schema.target_column and schema.target_column in df.columns:
        cols.append(schema.target_column)

    index_df = df[cols].copy()
    index_df["joined_text"] = texts
    return index_df


def fit_and_save_global_pca(
    *,
    train_embeddings: np.ndarray,
    dev_embeddings: np.ndarray | None,
    dims: list[int],
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    pca_info: dict[str, Any] = {}

    for dim in dims:
        n_components = min(dim, train_embeddings.shape[0], train_embeddings.shape[1])
        if n_components < 2:
            continue

        print(f"Fitting global PCA with n_components={n_components}")
        pca = PCA(n_components=n_components, random_state=42)

        train_pca = pca.fit_transform(train_embeddings)
        np.save(output_dir / f"train_{prefix}_pca{n_components}.npy", train_pca)

        if dev_embeddings is not None:
            dev_pca = pca.transform(dev_embeddings)
            np.save(output_dir / f"dev_{prefix}_pca{n_components}.npy", dev_pca)

        joblib.dump(pca, output_dir / f"{prefix}_pca{n_components}.joblib")

        pca_info[f"pca_{n_components}"] = {
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "train_file": f"train_{prefix}_pca{n_components}.npy",
            "dev_file": f"dev_{prefix}_pca{n_components}.npy" if dev_embeddings is not None else None,
            "pca_object_file": f"{prefix}_pca{n_components}.joblib",
        }

    return pca_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and save BGE embeddings.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg.outputs.output_dir)

    print(f"Loading train file: {cfg.inputs.train_file}")
    train_df = pd.read_csv(cfg.inputs.train_file)
    train_texts = build_texts(
        train_df,
        text_columns=cfg.text.text_columns,
        text_join_mode=cfg.text.text_join_mode,
    )

    dev_df: pd.DataFrame | None = None
    dev_texts: list[str] | None = None
    if cfg.inputs.dev_file:
        print(f"Loading dev file: {cfg.inputs.dev_file}")
        dev_df = pd.read_csv(cfg.inputs.dev_file)
        dev_texts = build_texts(
            dev_df,
            text_columns=cfg.text.text_columns,
            text_join_mode=cfg.text.text_join_mode,
        )

    print(f"Loading encoder: {cfg.embeddings.model_name}")
    encoder = SentenceTransformer(cfg.embeddings.model_name)

    print("Encoding train texts...")
    train_embeddings = encoder.encode(
        train_texts,
        batch_size=cfg.embeddings.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.embeddings.normalize_embeddings,
    ).astype(np.float32)

    np.save(output_dir / f"train_{cfg.outputs.prefix}_embeddings.npy", train_embeddings)

    train_index_df = build_index_df(train_df, schema=cfg.schema, texts=train_texts)
    train_index_df.to_csv(output_dir / "train_embedding_index.csv", index=False)

    dev_embeddings: np.ndarray | None = None
    if dev_df is not None and dev_texts is not None:
        print("Encoding dev texts...")
        dev_embeddings = encoder.encode(
            dev_texts,
            batch_size=cfg.embeddings.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=cfg.embeddings.normalize_embeddings,
        ).astype(np.float32)

        np.save(output_dir / f"dev_{cfg.outputs.prefix}_embeddings.npy", dev_embeddings)

        dev_index_df = build_index_df(dev_df, schema=cfg.schema, texts=dev_texts)
        dev_index_df.to_csv(output_dir / "dev_embedding_index.csv", index=False)

    manifest: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "encoder_name": cfg.embeddings.model_name,
        "normalize_embeddings": cfg.embeddings.normalize_embeddings,
        "text_columns": cfg.text.text_columns,
        "text_join_mode": cfg.text.text_join_mode,
        "train_embeddings_file": f"train_{cfg.outputs.prefix}_embeddings.npy",
        "dev_embeddings_file": f"dev_{cfg.outputs.prefix}_embeddings.npy" if dev_embeddings is not None else None,
        "train_index_file": "train_embedding_index.csv",
        "dev_index_file": "dev_embedding_index.csv" if dev_embeddings is not None else None,
        "train_shape": list(train_embeddings.shape),
        "dev_shape": list(dev_embeddings.shape) if dev_embeddings is not None else None,
        "pca": {},
    }

    dims = cfg.pca.dimensions or [128, 256]
    if cfg.pca.save_global_pca:
        manifest["pca"] = fit_and_save_global_pca(
            train_embeddings=train_embeddings,
            dev_embeddings=dev_embeddings,
            dims=dims,
            output_dir=output_dir,
            prefix=cfg.outputs.prefix,
        )

    save_json(manifest, output_dir / "embedding_manifest.json")
    print(f"Saved embedding manifest to: {output_dir / 'embedding_manifest.json'}")


if __name__ == "__main__":
    main()