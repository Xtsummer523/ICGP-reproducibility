from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import MODELS, RESULTS, SimConfig, TrainConfig
from .dataset import FEATURE_DIM, TARGET_DIM, PredictionDataset, generate_prediction_arrays
from .models import ResidualIntentTrajectoryMLP, TrajectoryMLP


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics(pred: torch.Tensor, target: torch.Tensor, horizon: int, neighbor_limit: int) -> tuple[float, float]:
    p = pred.detach().cpu().numpy().reshape(-1, horizon, neighbor_limit, 2) * 6.0
    y = target.detach().cpu().numpy().reshape(-1, horizon, neighbor_limit, 2) * 6.0
    valid = np.linalg.norm(y, axis=-1) > 1.0e-6
    err = np.linalg.norm(p - y, axis=-1)
    ade = float(err[valid].mean()) if np.any(valid) else 0.0
    final_valid = valid[:, -1, :]
    fde = float(err[:, -1, :][final_valid].mean()) if np.any(final_valid) else 0.0
    return ade, fde


def train_models(sim_cfg: SimConfig, train_cfg: TrainConfig) -> tuple[TrajectoryMLP, TrajectoryMLP, pd.DataFrame, pd.DataFrame]:
    _set_seed(train_cfg.seed)
    total = train_cfg.train_samples + train_cfg.val_samples + train_cfg.test_samples
    arrays = generate_prediction_arrays(sim_cfg, total, seed=train_cfg.seed)
    n_train = train_cfg.train_samples
    n_val = train_cfg.val_samples
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, total),
    }
    np.savez_compressed(RESULTS / "prediction_dataset.npz", **arrays)

    histories = []
    test_rows = []
    models = {}
    for name, x_key in [("intent", "x_intent"), ("passive", "x_passive")]:
        if name == "intent":
            model = ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
        else:
            model = TrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1.0e-4)
        loss_fn = nn.SmoothL1Loss()
        train_ds = PredictionDataset(arrays[x_key][splits["train"]], arrays["y"][splits["train"]])
        val_x = torch.from_numpy(arrays[x_key][splits["val"]])
        val_y = torch.from_numpy(arrays["y"][splits["val"]])
        loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=False)
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(loss_fn(val_pred, val_y).item())
                val_ade, val_fde = _metrics(val_pred, val_y, sim_cfg.horizon, sim_cfg.neighbor_limit)
            histories.append(
                {
                    "model": name,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_loss": val_loss,
                    "val_ADE": val_ade,
                    "val_FDE": val_fde,
                }
            )
        test_x = torch.from_numpy(arrays[x_key][splits["test"]])
        test_y = torch.from_numpy(arrays["y"][splits["test"]])
        with torch.no_grad():
            test_pred = model(test_x)
            test_ade, test_fde = _metrics(test_pred, test_y, sim_cfg.horizon, sim_cfg.neighbor_limit)
            test_loss = float(loss_fn(test_pred, test_y).item())
        test_rows.append({"model": name, "test_loss": test_loss, "ADE": test_ade, "FDE": test_fde})
        torch.save(model.state_dict(), MODELS / f"{name}_predictor.pt")
        models[name] = model

    hist_df = pd.DataFrame(histories)
    test_df = pd.DataFrame(test_rows)
    hist_df.to_csv(RESULTS / "training_history.csv", index=False)
    test_df.to_csv(RESULTS / "prediction_test_metrics.csv", index=False)
    with open(RESULTS / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({"sim": asdict(sim_cfg), "train": asdict(train_cfg)}, f, indent=2)
    return models["intent"], models["passive"], hist_df, test_df
