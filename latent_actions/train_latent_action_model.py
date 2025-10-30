#!/usr/bin/env python3
"""Streamed training for latent action regression models.

The trainer repeatedly samples small, random subsets of rides, builds an
in-memory dataset from those rides, fits a regression head that predicts the
continuous action vector (e.g. linear/angular velocities), and then repeats with
fresh data. This limits peak memory usage while still touching the full corpus
over time. Metrics are logged to Weights & Biases if configured.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:  # pragma: no cover - handled at runtime when wandb missing
    wandb = None  # type: ignore


# ---------------------------------------------------------------------------
# Data utilities


def read_action_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Action CSV at {path} is missing a header row.")
        if "frame_id" not in reader.fieldnames:
            raise ValueError(f"'frame_id' column missing in {path}")
        action_columns = [col for col in reader.fieldnames if col != "frame_id"]
        if not action_columns:
            raise ValueError(f"No action columns found in {path}")
        frame_ids: List[int] = []
        actions: List[List[float]] = []
        for row in reader:
            try:
                frame_ids.append(int(row["frame_id"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame_id value '{row['frame_id']}' in {path}") from exc
            try:
                actions.append([float(row[col]) for col in action_columns])
            except ValueError as exc:
                raise ValueError(f"Non-numeric action value in {path}: {row}") from exc
    if not frame_ids:
        raise ValueError(f"No rows parsed from action CSV {path}")
    order = np.argsort(frame_ids)
    sorted_actions = np.asarray(actions, dtype=np.float32)[order]
    sorted_frames = np.asarray(frame_ids, dtype=np.int64)[order]
    return sorted_frames, sorted_actions


def find_action_files(actions_root: Path, split: str, action_kind: str, ride_ids: Optional[Sequence[str]]) -> List[Path]:
    split_root = actions_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Actions split directory not found: {split_root}")
    pattern = f"{action_kind}_actions_*.csv"
    candidates = sorted(split_root.glob("output_rides_*/*/" + pattern))
    if ride_ids:
        ride_filters = set(ride_ids)
        candidates = [path for path in candidates if path.parent.name in ride_filters]
        if not candidates:
            raise ValueError(f"No action files matched the requested ride ids: {ride_ids}")
    if not candidates:
        raise FileNotFoundError(f"No action CSVs matching pattern '{pattern}' found under {split_root}")
    return candidates


def load_tensor(encoded_path: Path) -> torch.Tensor:
    tensor = torch.load(encoded_path, map_location="cpu")
    if isinstance(tensor, dict):
        for value in tensor.values():
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        else:
            raise ValueError(f"No tensor found inside dictionary at {encoded_path}")
    if isinstance(tensor, (list, tuple)):
        if not tensor:
            raise ValueError(f"Encoded tensor list is empty at {encoded_path}")
        tensor = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in tensor])
    if tensor.dim() == 5 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 4 and tensor.shape[0] <= 64 and tensor.shape[1] > tensor.shape[0]:
        tensor = tensor.permute(1, 0, 2, 3)
    if tensor.dim() < 3:
        raise ValueError(f"Encoded tensor must have at least 3 dimensions, got shape {tuple(tensor.shape)} at {encoded_path}")
    return tensor.float()


WINDOW_SIZE = 8
ACTION_INDEX = WINDOW_SIZE - 2  # Predict action taken at frame index 6 (0-based)


# ---------------------------------------------------------------------------
# Dataset and streaming loader


@dataclass
class RideData:
    ride_name: str
    encoded_dir: Path
    action_csv: Path


class TransitionDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def collect_rides(actions_root: Path, encoded_root: Path, split: str, action_kind: str, ride_ids: Optional[Sequence[str]]) -> List[RideData]:
    action_paths = find_action_files(actions_root, split, action_kind, ride_ids)
    rides: List[RideData] = []
    for csv_path in action_paths:
        ride_dir = csv_path.parent
        ride_name = ride_dir.name
        relative = ride_dir.relative_to(actions_root / split)
        encoded_dir = encoded_root / split / relative
        rides.append(RideData(ride_name=ride_name, encoded_dir=encoded_dir, action_csv=csv_path))
    return rides


def prepare_ride_samples(ride: RideData, pool_hw: int, action_dim: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    feature_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []

    encoded_files = sorted(ride.encoded_dir.glob("encoded_video_*.pt"))
    if not encoded_files:
        return feature_list, target_list

    frame_segments: List[torch.Tensor] = []
    for encoded_file in encoded_files:
        frames = load_tensor(encoded_file)
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        frame_segments.append(frames)

    if not frame_segments:
        return feature_list, target_list

    frames = torch.cat(frame_segments, dim=0).float()  # (T, C, H, W)
    if pool_hw > 0:
        frames = F.adaptive_avg_pool2d(frames, (pool_hw, pool_hw))

    _, action_values = read_action_csv(ride.action_csv)
    total_frames = frames.shape[0]
    if total_frames < WINDOW_SIZE or action_values.shape[0] <= ACTION_INDEX:
        return feature_list, target_list

    max_windows = total_frames - WINDOW_SIZE + 1
    max_actions = action_values.shape[0] - ACTION_INDEX
    windows = min(max_windows, max_actions)
    if windows <= 0:
        return feature_list, target_list

    for start in range(windows):
        window_frames = frames[start : start + WINDOW_SIZE]  # (T, C, H, W)
        window_tensor = window_frames.permute(1, 0, 2, 3).contiguous().clone()  # (C, T, H, W)
        feature_list.append(window_tensor)
        action_vec = action_values[start + ACTION_INDEX]
        if action_vec.shape[0] != action_dim:
            raise ValueError(
                f"Action dimension mismatch for ride {ride.ride_name}: expected {action_dim}, got {action_vec.shape[0]}"
            )
        target_list.append(torch.from_numpy(action_vec.copy()).float())

    return feature_list, target_list


def load_chunk_dataset(
    rides: Sequence[RideData],
    pool_hw: int,
    action_dim: int,
    num_workers: int,
) -> TransitionDataset:
    feature_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []

    if num_workers <= 1:
        for ride in rides:
            ride_features, ride_targets = prepare_ride_samples(ride, pool_hw, action_dim)
            feature_list.extend(ride_features)
            target_list.extend(ride_targets)
    else:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(rides))) as executor:
            futures = [executor.submit(prepare_ride_samples, ride, pool_hw, action_dim) for ride in rides]
            for future in futures:
                ride_features, ride_targets = future.result()
                feature_list.extend(ride_features)
                target_list.extend(ride_targets)

    if not feature_list:
        raise RuntimeError("Chunk loader failed to produce any samples; check that encoded videos exist for selected rides.")

    c_sizes = [feat.shape[0] for feat in feature_list]
    t_sizes = [feat.shape[1] for feat in feature_list]
    h_sizes = [feat.shape[2] for feat in feature_list]
    w_sizes = [feat.shape[3] for feat in feature_list]

    max_c = max(c_sizes)
    max_t = max(t_sizes)
    max_h = max(h_sizes)
    max_w = max(w_sizes)

    features = torch.zeros(len(feature_list), max_c, max_t, max_h, max_w, dtype=torch.float32)
    for idx, feat in enumerate(feature_list):
        c, t, h, w = feat.shape
        features[idx, :c, :t, :h, :w] = feat

    targets = torch.stack(target_list).float()
    return TransitionDataset(features, targets)


def chunk_iterator(rides: Sequence[RideData], chunk_size: int, infinite: bool = True) -> Iterator[List[RideData]]:
    indices = list(range(len(rides)))
    while True:
        random.shuffle(indices)
        for start in range(0, len(indices), chunk_size):
            chunk = [rides[idx] for idx in indices[start : start + chunk_size]]
            yield chunk
        if not infinite:
            break


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    rides: Sequence[RideData],
    args: argparse.Namespace,
    action_dim: int,
    split_name: str,
) -> Optional[Dict[str, float]]:
    if not rides:
        print(f"No rides provided for {split_name} evaluation.")
        return None

    eval_chunk = args.eval_chunk_size if args.eval_chunk_size > 0 else args.chunk_size
    total_mse = 0.0
    total_mae = 0.0
    total_vectors = 0
    total_samples = 0

    previous_mode = model.training
    model.eval()

    for start in range(0, len(rides), eval_chunk):
        batch_rides = rides[start : start + eval_chunk]
        dataset = load_chunk_dataset(batch_rides, args.pool_size, action_dim, args.num_workers)
        features = dataset.features.to(device)
        targets = dataset.targets.to(device)
        with torch.no_grad():
            preds = model(features)
            mse_sum = F.mse_loss(preds, targets, reduction="sum").item()
            mae_sum = F.l1_loss(preds, targets, reduction="sum").item()
        total_mse += mse_sum
        total_mae += mae_sum
        total_vectors += targets.numel()
        total_samples += targets.shape[0]

        del dataset, features, targets

    if previous_mode:
        model.train()

    mean_mse = total_mse / max(1, total_vectors)
    mean_mae = total_mae / max(1, total_vectors)
    print(
        f"{split_name.capitalize()} evaluation | samples={total_samples} | "
        f"mse={mean_mse:.6f} | mae={mean_mae:.6f}"
    )
    return {
        "mse": mean_mse,
        "mae": mean_mae,
        "samples": float(total_samples),
        "vectors": float(total_vectors),
    }


# ---------------------------------------------------------------------------
# Model and training


class Action3DCNN(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        mlp_layers: List[nn.Module] = []
        prev_dim = 256
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, action_dim))
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


def infer_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def scale_for_gpu(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        return
    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    total_mem_gb = []
    gpu_names = []
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb.append(props.total_memory / (1024 ** 3))
        gpu_names.append(props.name)

    if not total_mem_gb:
        return

    min_mem = min(total_mem_gb)
    max_mem = max(total_mem_gb)
    canonical_name = gpu_names[0].upper()

    scaled = False
    if "H100" in canonical_name:
        target_chunk = max(args.chunk_size, 32)
        target_batch = max(args.batch_size, 2048)
        if target_chunk != args.chunk_size or target_batch != args.batch_size:
            args.chunk_size = target_chunk
            args.batch_size = target_batch
            scaled = True
    else:
        baseline_mem = 40.0  # GB reference (roughly A100 40GB)
        scale_factor = max(1.0, min_mem / baseline_mem)
        scaled_chunk = max(args.chunk_size, int(math.ceil(args.chunk_size * scale_factor)))
        scaled_batch = max(args.batch_size, int(math.ceil(args.batch_size * scale_factor)))
        if scaled_chunk != args.chunk_size or scaled_batch != args.batch_size:
            args.chunk_size = scaled_chunk
            args.batch_size = scaled_batch
            scaled = True

    if scaled:
        mem_str = ", ".join(f"{mem:.1f}GB" for mem in total_mem_gb)
        print(
            f"Auto-scaled training sizes for GPUs [{mem_str}] "
            f"(detected {gpu_names[0]}): chunk_size={args.chunk_size}, batch_size={args.batch_size}"
        )


# ---------------------------------------------------------------------------
# Checkpoint utilities


def save_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str,
    model_state: Dict[str, torch.Tensor],
    optimizer_state: Optional[Dict] = None,
    action_dim: int = 0,
    input_channels: int = 0,
    args: Optional[argparse.Namespace] = None,
    best_mse: float = float("inf"),
    total_chunks: int = 0,
    global_step: int = 0,
    train_metrics: Optional[Dict[str, float]] = None,
    eval_metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Save a checkpoint with model and training state."""
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "model_state": model_state,
        "action_dim": action_dim,
        "input_channels": input_channels,
        "best_mse": best_mse,
        "total_chunks": total_chunks,
        "global_step": global_step,
    }
    
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    
    if args is not None:
        payload.update({
            "hidden_dims": list(args.hidden_dims),
            "pool_size": args.pool_size,
            "action_kind": args.action_kind,
            "metadata": {
                "split": args.split,
                "chunk_size": args.chunk_size,
                "chunk_epochs": args.chunk_epochs,
                "patience": args.patience,
                "target_mse": args.target_mse,
                "best_mse": best_mse,
                "total_chunks": total_chunks,
                "eval_split": args.eval_split,
            },
        })
    
    if train_metrics or eval_metrics:
        payload["evaluation"] = {
            "train": train_metrics if train_metrics else {},
            "eval": eval_metrics if eval_metrics else {},
        }
    
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a checkpoint from disk."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def train_streaming(
    args: argparse.Namespace,
    rides: Sequence[RideData],
    action_dim: int,
) -> None:
    device = infer_device(args.device)
    model = None
    optimizer: Optional[torch.optim.Optimizer] = None
    criterion = nn.MSELoss()
    
    # Setup checkpoint directories
    checkpoint_base = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    best_checkpoint_dir = checkpoint_base / "best" if checkpoint_base else None
    current_checkpoint_dir = checkpoint_base / "current" if checkpoint_base else None
    
    # Initialize timing for periodic saves (every 15 minutes = 900 seconds)
    last_checkpoint_time = time.time()
    checkpoint_interval = 900  # 15 minutes
    best_state_updated = False  # Track if best state has changed since last save

    if args.eval_split:
        eval_ride_ids = set(args.eval_ride_ids) if args.eval_ride_ids else None
        eval_rides = collect_rides(
            actions_root=args.actions_root,
            encoded_root=args.encoded_root,
            split=args.eval_split,
            action_kind=args.action_kind,
            ride_ids=eval_ride_ids,
        )
    else:
        eval_rides = []

    run = None
    if args.wandb_project:
        if wandb is None:
            raise RuntimeError("wandb is not installed but wandb logging is requested.")
        wandb_config = {
            **vars(args),
            "num_rides": len(rides),
            "action_dim": action_dim,
            "eval_split": args.eval_split,
            "eval_rides": len(eval_rides),
        }
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=wandb_config,
        )

    chunk_gen = chunk_iterator(rides, args.chunk_size, infinite=True)
    total_chunks = 0
    consecutive_hits = 0
    best_mse = float("inf")
    global_step = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    input_channels: Optional[int] = None
    train_metrics: Optional[Dict[str, float]] = None
    eval_metrics: Optional[Dict[str, float]] = None

    try:
        while True:
            if args.max_chunks and total_chunks >= args.max_chunks:
                print(f"Reached max_chunks={args.max_chunks}, stopping.")
                break

            chunk_rides = next(chunk_gen)
            chunk_names = [ride.ride_name for ride in chunk_rides]
            print(f"\n=== Loading chunk #{total_chunks + 1} with rides: {chunk_names} ===")
            dataset = load_chunk_dataset(chunk_rides, args.pool_size, action_dim, args.num_workers)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=device.type == "cuda",
            )

            if model is None:
                input_channels = dataset.features.shape[1]
                model = Action3DCNN(input_channels, action_dim, args.hidden_dims, dropout=args.dropout).to(device)
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif input_channels is None:
                input_channels = dataset.features.shape[1]

            chunk_steps = 0
            for epoch in range(args.chunk_epochs):
                for features, targets in loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()  # type: ignore
                    logits = model(features)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()  # type: ignore
                    chunk_steps += 1
                    global_step += 1

            with torch.no_grad():
                model.eval()
                features = dataset.features.to(device)
                targets = dataset.targets.to(device)
                logits = model(features)
                mse = criterion(logits, targets).item()
                mae = torch.mean(torch.abs(logits - targets)).item()
                model.train()

            if mse < best_mse or best_state is None:
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                best_state = base_model.state_dict()
                best_mse = mse
                best_state_updated = True  # Mark that best state has changed
            total_chunks += 1
            if args.target_mse > 0:
                consecutive_hits = consecutive_hits + 1 if mse <= args.target_mse else 0
            else:
                consecutive_hits = 0

            print(
                f"Chunk {total_chunks} | samples={len(dataset)} | "
                f"mse={mse:.6f} | mae={mae:.6f} | "
                f"consecutive_hits={consecutive_hits}"
            )

            if run:
                wandb.log(
                    {
                        "chunk": total_chunks,
                        "chunk_mse": mse,
                        "chunk_mae": mae,
                        "best_mse": best_mse,
                        "consecutive_hits": consecutive_hits,
                        "samples": len(dataset),
                    },
                    step=global_step,
                )

            # Periodic checkpoint saving (every 10 minutes)
            current_time = time.time()
            if checkpoint_base and (current_time - last_checkpoint_time) >= checkpoint_interval:
                if model is not None and best_state is not None and input_channels is not None:
                    base_model = model.module if isinstance(model, nn.DataParallel) else model
                    current_state = base_model.state_dict()
                    
                    # Save current checkpoint
                    checkpoint_name = f"{args.action_kind}_actions_current.pt"
                    save_checkpoint(
                        checkpoint_dir=current_checkpoint_dir,
                        checkpoint_name=checkpoint_name,
                        model_state=current_state,
                        optimizer_state=optimizer.state_dict() if optimizer else None,
                        action_dim=action_dim,
                        input_channels=input_channels,
                        args=args,
                        best_mse=mse,  # Current MSE
                        total_chunks=total_chunks,
                        global_step=global_step,
                    )
                    print(f"✓ Saved current checkpoint to {current_checkpoint_dir / checkpoint_name}")
                    
                    # Save best checkpoint only if it has been updated since last save
                    if best_state_updated:
                        checkpoint_name = f"{args.action_kind}_actions_best.pt"
                        save_checkpoint(
                            checkpoint_dir=best_checkpoint_dir,
                            checkpoint_name=checkpoint_name,
                            model_state=best_state,
                            optimizer_state=None,  # Don't need optimizer state for best
                            action_dim=action_dim,
                            input_channels=input_channels,
                            args=args,
                            best_mse=best_mse,
                            total_chunks=total_chunks,
                            global_step=global_step,
                        )
                        print(f"✓ Saved best checkpoint to {best_checkpoint_dir / checkpoint_name}")
                        best_state_updated = False  # Reset flag after saving
                    else:
                        print(f"⊘ Skipped best checkpoint (no improvement since last save)")
                    
                    last_checkpoint_time = current_time

            if args.target_mse > 0 and consecutive_hits >= args.patience:
                print(
                    f"Target MSE {args.target_mse:.6f} reached for "
                    f"{consecutive_hits} consecutive chunks. Stopping training."
                )
                break

        if best_state is None or input_channels is None:
            raise RuntimeError("Training did not produce a valid model state.")

        eval_model = Action3DCNN(input_channels, action_dim, args.hidden_dims, dropout=args.dropout).to(device)
        eval_model.load_state_dict(best_state)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            eval_model = nn.DataParallel(eval_model)

        train_metrics = evaluate_model(eval_model, device, rides, args, action_dim, "train")
        eval_metrics = evaluate_model(eval_model, device, eval_rides, args, action_dim, args.eval_split or "eval") if eval_rides else None

        if run:
            summary_log = {
                "best_mse": best_mse,
            }
            if train_metrics:
                summary_log.update(
                    {
                        "train_mse": train_metrics["mse"],
                        "train_mae": train_metrics["mae"],
                    }
                )
            if eval_metrics:
                summary_log.update(
                    {
                        "eval_mse": eval_metrics["mse"],
                        "eval_mae": eval_metrics["mae"],
                    }
                )
            wandb.log(summary_log, step=global_step)

    finally:
        if run:
            run.finish()
        
        # Save final checkpoints
        if best_state is not None and input_channels is not None:
            # Save to new checkpoint directory structure
            if checkpoint_base:
                print("\nSaving final checkpoints...")
                
                # Save best checkpoint
                checkpoint_name = f"{args.action_kind}_actions_best.pt"
                save_checkpoint(
                    checkpoint_dir=best_checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                    model_state=best_state,
                    optimizer_state=None,
                    action_dim=action_dim,
                    input_channels=input_channels,
                    args=args,
                    best_mse=best_mse,
                    total_chunks=total_chunks,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )
                print(f"✓ Final best checkpoint saved to {best_checkpoint_dir / checkpoint_name}")
                
                # Save current checkpoint if model exists
                if model is not None:
                    base_model = model.module if isinstance(model, nn.DataParallel) else model
                    current_state = base_model.state_dict()
                    checkpoint_name = f"{args.action_kind}_actions_current.pt"
                    save_checkpoint(
                        checkpoint_dir=current_checkpoint_dir,
                        checkpoint_name=checkpoint_name,
                        model_state=current_state,
                        optimizer_state=optimizer.state_dict() if optimizer else None,
                        action_dim=action_dim,
                        input_channels=input_channels,
                        args=args,
                        best_mse=best_mse,
                        total_chunks=total_chunks,
                        global_step=global_step,
                        train_metrics=train_metrics,
                        eval_metrics=eval_metrics,
                    )
                    print(f"✓ Final current checkpoint saved to {current_checkpoint_dir / checkpoint_name}")
            
            # Maintain backwards compatibility with old checkpoint_path argument
            if args.checkpoint_path:
                path = args.checkpoint_path
                if path.exists():
                    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                    path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
                    print(f"Checkpoint path exists; writing new checkpoint to {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model_state": best_state,
                    "hidden_dims": list(args.hidden_dims),
                    "pool_size": args.pool_size,
                    "action_kind": args.action_kind,
                    "action_dim": action_dim,
                    "metadata": {
                        "split": args.split,
                        "chunk_size": args.chunk_size,
                        "chunk_epochs": args.chunk_epochs,
                        "patience": args.patience,
                        "target_mse": args.target_mse,
                        "best_mse": best_mse,
                        "total_chunks": total_chunks,
                        "eval_split": args.eval_split,
                    },
                    "evaluation": {
                        "train": train_metrics if train_metrics else {},
                        "eval": eval_metrics if eval_metrics else {},
                    },
                }
                torch.save(payload, path)
                print(f"Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Argument parsing / entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded-root", type=Path, required=True, help="Root directory that contains encoded frame tensors.")
    parser.add_argument("--actions-root", type=Path, required=True, help="Root directory that contains action CSV files.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g. train, test).")
    parser.add_argument("--action-kind", type=str, choices=["input", "output"], default="input", help="Which action CSVs to use.")
    parser.add_argument("--ride-ids", type=str, nargs="*", default=None, help="Optional list of ride directory names to include.")
    parser.add_argument("--pool-size", type=int, default=32, help="Spatial size for adaptive avg pooling before 3D encoding (set 0 to disable).")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(256, 128), help="Hidden layer sizes for the MLP.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability applied after hidden layers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training within each chunk.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Number of rides to include per streamed chunk.")
    parser.add_argument("--chunk-epochs", type=int, default=1, help="Number of epochs to run on each chunk before sampling new rides.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap on total chunks processed (0 means unlimited).")
    parser.add_argument("--patience", type=int, default=3, help="Stop once target MSE is met for this many consecutive chunks.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam.")
    parser.add_argument("--target-mse", type=float, default=0.0, help="Optional target MSE for early stopping (<=0 disables early stop).")
    parser.add_argument("--device", type=str, default="auto", help="Training device ('cpu', 'cuda', or 'auto').")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of worker threads for parallel data loading per chunk.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional path to save the trained model state (legacy format).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for periodic checkpoints (creates 'best' and 'current' subdirs).")
    parser.add_argument("--eval-split", type=str, default="", help="Optional split name for evaluation (e.g. 'test').")
    parser.add_argument("--eval-ride-ids", type=str, nargs="*", default=None, help="Optional specific rides for evaluation split.")
    parser.add_argument("--eval-chunk-size", type=int, default=0, help="Chunk size for evaluation (0 reuses chunk-size).")

    # Weights & Biases logging
    parser.add_argument("--wandb-project", type=str, default="", help="W&B project name (set to enable logging).")
    parser.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity/team.")
    parser.add_argument("--wandb-run-name", type=str, default="", help="Optional W&B run name.")
    parser.add_argument("--wandb-group", type=str, default="", help="Optional W&B group.")
    parser.add_argument("--wandb-mode", type=str, default="online", help="W&B mode: online, offline, or disabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale_for_gpu(args)

    rides = collect_rides(
        actions_root=args.actions_root,
        encoded_root=args.encoded_root,
        split=args.split,
        action_kind=args.action_kind,
        ride_ids=args.ride_ids,
    )
    if len(rides) == 0:
        raise RuntimeError("No rides found for the given configuration.")

    sample_actions = read_action_csv(rides[0].action_csv)[1]
    action_dim = sample_actions.shape[1]
    print(f"Detected action dimension: {action_dim}")
    print(
        f"Window size: {WINDOW_SIZE} frames (predict action index {ACTION_INDEX}) | spatial pool size: {args.pool_size}"
    )

    train_streaming(args, rides, action_dim)


if __name__ == "__main__":
    main()
