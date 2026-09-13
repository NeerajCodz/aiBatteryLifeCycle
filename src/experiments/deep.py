"""Battery-grouped sequence-model evaluation with train-only preprocessing."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import regression_metrics
from src.evaluation.protocol import grouped_train_val_test_folds


TORCH_MODEL_IDS = (
    "vanilla_lstm", "bidirectional_lstm", "gru", "attention_lstm",
    "battery_gpt", "temporal_fusion_transformer", "vae_lstm",
)
TF_MODEL_IDS = ("itransformer", "physics_itransformer", "dynamic_graph_itransformer")


class SequenceStandardizer:
    """Median-impute and standardize sequence channels using training rows only."""

    def fit(self, X: np.ndarray) -> "SequenceStandardizer":
        flat = np.asarray(X, dtype=float).reshape(-1, X.shape[-1])
        self.median_ = np.nanmedian(flat, axis=0)
        filled = np.where(np.isnan(flat), self.median_, flat)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        filled = np.where(np.isnan(values), self.median_, values)
        return ((filled - self.mean_) / self.scale_).astype(np.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _torch_factories(n_features: int) -> dict[str, Callable[[], object]]:
    from src.models.deep.lstm import AttentionLSTM, BidirectionalLSTM, GRUModel, VanillaLSTM
    from src.models.deep.transformer import BatteryGPT, TemporalFusionTransformer
    from src.models.deep.vae_lstm import VAE_LSTM

    return {
        "vanilla_lstm": lambda: VanillaLSTM(n_features, hidden_dim=64, n_layers=2),
        "bidirectional_lstm": lambda: BidirectionalLSTM(n_features, hidden_dim=64, n_layers=2),
        "gru": lambda: GRUModel(n_features, hidden_dim=64, n_layers=2),
        "attention_lstm": lambda: AttentionLSTM(n_features, hidden_dim=64, n_layers=2),
        "battery_gpt": lambda: BatteryGPT(n_features, d_model=64, n_heads=4, n_layers=2),
        "temporal_fusion_transformer": lambda: TemporalFusionTransformer(
            n_features, d_model=64, n_heads=4, n_layers=2
        ),
        "vae_lstm": lambda: VAE_LSTM(
            n_features, seq_len=64, hidden_dim=64, latent_dim=16, n_layers=2
        ),
    }


def _train_torch_model(
    model_id: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.deep.lstm import train_loop
    from src.models.deep.vae_lstm import train_vae

    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    normalize = lambda values: ((values - y_mean) / y_scale).astype(np.float32)
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(normalize(y_train))
    ), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(normalize(y_val))
    ), batch_size=batch_size, shuffle=False)
    if model_id == "vae_lstm":
        history = train_vae(
            model, train_loader, val_loader, max_epochs=max_epochs,
            patience=patience, device=device, warmup_epochs=min(30, max_epochs // 3),
        )
    else:
        history = train_loop(
            model, train_loader, val_loader, max_epochs=max_epochs,
            patience=patience, device=device,
        )
    model.eval()

    def predict(values: np.ndarray) -> np.ndarray:
        outputs = []
        loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
        with torch.no_grad():
            for (xb,) in loader:
                raw = model(xb.to(device))
                if isinstance(raw, dict):
                    raw = raw["health_pred"]
                outputs.append(raw.detach().cpu().numpy().reshape(-1))
        return np.concatenate(outputs) * y_scale + y_mean

    epochs = len(history.get("train_losses", []))
    return predict(X_train), predict(X_test), epochs


def _train_tensorflow_model(
    model_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    import tensorflow as tf
    from src.models.deep.itransformer import (
        build_dynamic_graph_itransformer,
        build_itransformer,
        build_physics_itransformer,
    )

    builders = {
        "itransformer": build_itransformer,
        "physics_itransformer": build_physics_itransformer,
        "dynamic_graph_itransformer": build_dynamic_graph_itransformer,
    }
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    train_target = ((y_train - y_mean) / y_scale).astype(np.float32)
    val_target = ((y_val - y_mean) / y_scale).astype(np.float32)
    model = builders[model_id](X_train.shape[1], X_train.shape[2], d_model=32, n_heads=4, n_blocks=2)
    if model_id == "physics_itransformer":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={"soh_ml": "mae", "soh_phy": "mae"},
            loss_weights={"soh_ml": 1.0, "soh_phy": model.physics_loss_weight},
        )
        fit_y = {"soh_ml": train_target, "soh_phy": train_target}
        val_y = {"soh_ml": val_target, "soh_phy": val_target}
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
        fit_y, val_y = train_target, val_target
    history = model.fit(
        X_train, fit_y, validation_data=(X_val, val_y), epochs=max_epochs,
        batch_size=batch_size, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )],
    )

    def predict(values: np.ndarray) -> np.ndarray:
        raw = model.predict(values, batch_size=batch_size, verbose=0)
        if isinstance(raw, list):
            raw = raw[0]
        return np.asarray(raw).reshape(-1) * y_scale + y_mean

    result = (predict(X_train), predict(X_test), len(history.history["loss"]))
    tf.keras.backend.clear_session()
    return result


def run_grouped_sequence_benchmark(
    X: np.ndarray,
    index: pd.DataFrame,
    *,
    dataset_name: str,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (17, 42, 2026),
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    model_ids: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all ten sequence models without using test data for stopping."""
    if len(index) != len(X):
        raise ValueError("Sequence tensor and index have different row counts")
    requested = set(model_ids or (TORCH_MODEL_IDS + TF_MODEL_IDS))
    unknown = requested.difference(TORCH_MODEL_IDS + TF_MODEL_IDS)
    if unknown:
        raise KeyError(f"Unknown sequence model IDs: {sorted(unknown)}")
    y = index["SoH"].to_numpy(dtype=float)
    groups = index["battery_id"].astype(str).to_numpy()
    available_splits = min(n_splits, np.unique(groups).size)
    if available_splits < 2:
        raise ValueError("At least two batteries are required")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for seed in seeds:
        _set_seed(seed)
        folds = grouped_train_val_test_folds(
            groups, n_splits=available_splits, validation_fraction=0.2,
            random_state=seed,
        )
        for fold, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            scaler = SequenceStandardizer().fit(X[train_idx])
            X_train, X_val, X_test = (
                scaler.transform(X[subset]) for subset in (train_idx, val_idx, test_idx)
            )
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            for model_id, factory in _torch_factories(X.shape[-1]).items():
                if model_id not in requested:
                    continue
                print(
                    f"[{dataset_name}] seed={seed} fold={fold}/{available_splits} "
                    f"model={model_id} start",
                    flush=True,
                )
                _set_seed(seed)
                train_pred, test_pred, epochs = _train_torch_model(
                    model_id, factory(), X_train, y[train_idx], X_val, y[val_idx], X_test,
                    max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                    device=device,
                )
                _append_results(
                    metric_rows, prediction_rows, dataset_name, seed, fold, model_id,
                    train_idx, test_idx, groups, index, y, train_pred, test_pred, epochs,
                )
                print(
                    f"[{dataset_name}] seed={seed} fold={fold}/{available_splits} "
                    f"model={model_id} done epochs={epochs} mae={metric_rows[-1]['mae']:.4f}",
                    flush=True,
                )
            for model_id in TF_MODEL_IDS:
                if model_id not in requested:
                    continue
                print(
                    f"[{dataset_name}] seed={seed} fold={fold}/{available_splits} "
                    f"model={model_id} start",
                    flush=True,
                )
                _set_seed(seed)
                train_pred, test_pred, epochs = _train_tensorflow_model(
                    model_id, X_train, y[train_idx], X_val, y[val_idx], X_test,
                    max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                )
                _append_results(
                    metric_rows, prediction_rows, dataset_name, seed, fold, model_id,
                    train_idx, test_idx, groups, index, y, train_pred, test_pred, epochs,
                )
                print(
                    f"[{dataset_name}] seed={seed} fold={fold}/{available_splits} "
                    f"model={model_id} done epochs={epochs} mae={metric_rows[-1]['mae']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _append_results(
    metric_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    dataset_name: str,
    seed: int,
    fold: int,
    model_id: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
    index: pd.DataFrame,
    y: np.ndarray,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    epochs: int,
) -> None:
    metrics = regression_metrics(y[test_idx], test_pred)
    train_metrics = regression_metrics(y[train_idx], train_pred)
    metric_rows.append({
        "dataset": dataset_name, "seed": seed, "fold": fold, "model": model_id,
        **metrics, "train_mae": train_metrics["mae"],
        "generalization_gap_mae": metrics["mae"] - train_metrics["mae"],
        "epochs": epochs,
    })
    for local, row_idx in enumerate(test_idx):
        prediction_rows.append({
            "dataset": dataset_name, "seed": seed, "fold": fold, "model": model_id,
            "row_index": int(row_idx), "battery_id": groups[row_idx],
            "cycle_number": int(index.iloc[row_idx]["cycle_number"]),
            "y_true": y[row_idx], "y_pred": float(test_pred[local]),
            "residual": float(y[row_idx] - test_pred[local]),
        })


def run_zero_shot_sequence(
    source_X: np.ndarray,
    source_index: pd.DataFrame,
    targets: dict[str, tuple[np.ndarray, pd.DataFrame]],
    *,
    source_name: str = "NASA",
    seeds: tuple[int, ...] = (17, 42, 2026),
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    model_ids: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train each sequence model on NASA train/validation batteries once per seed.

    All target-domain labels remain untouched until scoring, and the same frozen
    source model is used for every supplied external target.
    """
    y_source = source_index["SoH"].to_numpy(dtype=float)
    source_groups = source_index["battery_id"].astype(str).to_numpy()
    lengths = {name: len(values[0]) for name, values in targets.items()}
    target_concat = np.concatenate([values[0] for values in targets.values()], axis=0)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    requested = set(model_ids or (TORCH_MODEL_IDS + TF_MODEL_IDS))
    unknown = requested.difference(TORCH_MODEL_IDS + TF_MODEL_IDS)
    if unknown:
        raise KeyError(f"Unknown sequence model IDs: {sorted(unknown)}")
    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(splitter.split(source_X, y_source, source_groups))
        scaler = SequenceStandardizer().fit(source_X[train_idx])
        X_train = scaler.transform(source_X[train_idx])
        X_val = scaler.transform(source_X[val_idx])
        X_target = scaler.transform(target_concat)
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        factories = _torch_factories(source_X.shape[-1])
        for model_id in TORCH_MODEL_IDS + TF_MODEL_IDS:
            if model_id not in requested:
                continue
            print(
                f"[{source_name}->external] seed={seed} model={model_id} start",
                flush=True,
            )
            _set_seed(seed)
            if model_id in factories:
                _, combined_pred, epochs = _train_torch_model(
                    model_id, factories[model_id](), X_train, y_source[train_idx],
                    X_val, y_source[val_idx], X_target, max_epochs=max_epochs,
                    patience=patience, batch_size=batch_size, device=device,
                )
            else:
                _, combined_pred, epochs = _train_tensorflow_model(
                    model_id, X_train, y_source[train_idx], X_val, y_source[val_idx],
                    X_target, max_epochs=max_epochs, patience=patience,
                    batch_size=batch_size,
                )
            offset = 0
            for target_name, (_, target_index) in targets.items():
                count = lengths[target_name]
                pred = combined_pred[offset : offset + count]
                offset += count
                y_target = target_index["SoH"].to_numpy(dtype=float)
                metric_rows.append({
                    "source_dataset": source_name,
                    "target_dataset": target_name,
                    "seed": seed,
                    "model": model_id,
                    "epochs": epochs,
                    **regression_metrics(y_target, pred),
                })
                for row, value in enumerate(pred):
                    prediction_rows.append({
                        "source_dataset": source_name,
                        "target_dataset": target_name,
                        "seed": seed,
                        "model": model_id,
                        "battery_id": str(target_index.iloc[row]["battery_id"]),
                        "cycle_number": int(target_index.iloc[row]["cycle_number"]),
                        "y_true": y_target[row],
                        "y_pred": float(value),
                        "residual": float(y_target[row] - value),
                    })
            print(
                f"[{source_name}->external] seed={seed} model={model_id} "
                f"done epochs={epochs}",
                flush=True,
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)
