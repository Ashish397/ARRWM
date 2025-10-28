#!/usr/bin/env python3
"""Train a latent action classifier that infers actions from consecutive states.

The script pairs encoded frame tensors (s_t, s_{t+1}) with either input or output
actions and trains a small MLP to predict the corresponding action class. It
expects the encoded tensors and action CSVs to share the same directory layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetSample:
    ride: str
    feature: torch.Tensor
    label: int


class TransitionDataset(Dataset):
    def __init__(self, samples: Sequence[DatasetSample]) -> None:
        self._features = torch.stack([s.feature for s in samples])
        self._labels = torch.tensor([s.label for s in samples], dtype=torch.long)
        self._rides = [s.ride for s in samples]

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._features[idx], self._labels[idx]

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def rides(self) -> List[str]:
        return list(self._rides)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded-root", type=Path, required=True, help="Root directory that contains encoded frame tensors.")
    parser.add_argument("--actions-root", type=Path, required=True, help="Root directory that contains action CSV files.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g. train, test).")
    parser.add_argument("--action-kind", type=str, choices=["input", "output"], default="input", help="Whether to use input_actions or output_actions CSVs.")
    parser.add_argument("--ride-ids", type=str, nargs="*", default=None, help="Optional list of ride directory names to include.")
    parser.add_argument("--pool-size", type=int, default=4, help="Spatial size for adaptive average pooling (per dimension).")
    parser.add_argument("--rounding", type=int, default=5, help="Number of decimals retained when grouping identical actions.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(256, 128), help="Hidden layer sizes for the MLP.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability applied after hidden layers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=2000, help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam.")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help="Required training accuracy threshold (use 1.0 to force perfect accuracy).",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional path to save the trained model state.")
    parser.add_argument("--device", type=str, default="auto", help="Training device ('cpu', 'cuda', or 'auto').")
    return parser.parse_args()


def infer_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


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
        # Some dumps may wrap the tensor under a specific key.
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


def encode_state(state: torch.Tensor, pool_hw: int) -> torch.Tensor:
    if state.dim() == 1:
        return state
    if state.dim() == 2:
        return state.flatten()
    if state.dim() == 3:
        pooled = F.adaptive_avg_pool2d(state, (pool_hw, pool_hw))
        return pooled.reshape(-1)
    raise ValueError(f"Unsupported state tensor with shape {tuple(state.shape)}")


def build_feature_triplet(state_t: torch.Tensor, state_tp1: torch.Tensor, pool_hw: int) -> torch.Tensor:
    feat_t = encode_state(state_t, pool_hw)
    feat_tp1 = encode_state(state_tp1, pool_hw)
    feat_delta = encode_state(state_tp1 - state_t, pool_hw)
    return torch.cat([feat_t, feat_tp1, feat_delta], dim=0)


def collect_samples(
    encoded_root: Path,
    actions_root: Path,
    split: str,
    action_kind: str,
    ride_ids: Optional[Sequence[str]],
    pool_hw: int,
    rounding: int,
) -> Tuple[TransitionDataset, torch.Tensor, List[str]]:
    action_csvs = find_action_files(actions_root, split, action_kind, ride_ids)
    samples: List[DatasetSample] = []
    rides_used: List[str] = []
    action_lookup: dict[Tuple[float, ...], int] = {}
    action_prototypes: List[np.ndarray] = []

    for action_csv in action_csvs:
        ride_dir = action_csv.parent
        ride_name = ride_dir.name
        relative = ride_dir.relative_to(actions_root / split)
        encoded_dir = encoded_root / split / relative

        encoded_files = sorted(encoded_dir.glob("encoded_video_*.pt"))
        if not encoded_files:
            raise FileNotFoundError(f"No encoded videos found for ride '{ride_name}' expected at {encoded_dir}")

        frame_ids, action_values = read_action_csv(action_csv)
        action_cursor = 0
        start_sample_count = len(samples)
        for encoded_file in encoded_files:
            frames = load_tensor(encoded_file)
            if frames.dim() not in (3, 4):
                raise ValueError(f"Unsupported encoded tensor rank {frames.dim()} at {encoded_file}")
            num_frames = frames.shape[0]
            if num_frames < 2:
                continue

            available_actions = action_values.shape[0] - action_cursor
            if available_actions <= 0:
                break

            max_pairs = min(num_frames - 1, available_actions)
            for idx in range(max_pairs):
                state_t = frames[idx]
                state_tp1 = frames[idx + 1]
                feature_vec = build_feature_triplet(state_t, state_tp1, pool_hw)
                action_vec = action_values[action_cursor + idx]
                class_id = assign_action_class(action_vec, action_lookup, action_prototypes, rounding)
                samples.append(DatasetSample(ride=ride_name, feature=feature_vec, label=class_id))
                rides_used.append(ride_name)
            action_cursor += max_pairs

        if len(samples) == start_sample_count:
            raise ValueError(f"No frames loaded for ride '{ride_name}'")

    if not samples:
        raise RuntimeError("No samples constructed from the provided data.")
    if not action_prototypes:
        raise RuntimeError("Failed to derive any unique actions from the provided CSV files.")

    dataset = TransitionDataset(samples)
    action_tensor = torch.tensor(action_prototypes, dtype=torch.float32)
    return dataset, action_tensor, rides_used


def assign_action_class(
    action_vec: np.ndarray,
    lookup: dict[Tuple[float, ...], int],
    prototypes: List[np.ndarray],
    rounding: int,
) -> int:
    key = tuple(np.round(action_vec, decimals=rounding).tolist())
    if key not in lookup:
        lookup[key] = len(prototypes)
        prototypes.append(action_vec.astype(np.float32))
    return lookup[key]


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


def train_model(
    dataset: TransitionDataset,
    action_values: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[MLPClassifier, float]:
    device = infer_device(args.device)
    model = MLPClassifier(dataset.features.shape[1], args.hidden_dims, action_values.shape[0], dropout=args.dropout)
    model.to(device)
    multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1
    if multi_gpu:
        model = nn.DataParallel(model)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    target_accuracy = args.min_accuracy
    eval_features = dataset.features.to(device)
    eval_labels = dataset.labels.to(device)
    history_loss: List[float] = []
    history_acc: List[float] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            logits = model(eval_features)
            predictions = logits.argmax(dim=1)
            correct = (predictions == eval_labels).float().mean().item()
            loss_value = criterion(logits, eval_labels).item()
            history_loss.append(loss_value)
            history_acc.append(correct)
            print(f"Epoch {epoch:04d} | loss={loss_value:.6f} | acc={correct * 100:.2f}%")
            if correct >= target_accuracy:
                print(f"Target accuracy {target_accuracy * 100:.2f}% reached at epoch {epoch}.")
                break
    else:
        print(f"Reached max epochs ({args.max_epochs}) with best accuracy {max(history_acc) * 100:.2f}%.")

    final_accuracy = history_acc[-1] if history_acc else 0.0
    if final_accuracy < target_accuracy:
        raise RuntimeError(f"Training accuracy {final_accuracy * 100:.2f}% did not reach required threshold {target_accuracy * 100:.2f}%.")

    trained_model = model.module if isinstance(model, nn.DataParallel) else model
    return trained_model, final_accuracy


def save_checkpoint(path: Path, model: nn.Module, action_values: torch.Tensor, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "action_values": action_values,
        "hidden_dims": list(args.hidden_dims),
        "pool_size": args.pool_size,
        "rounding": args.rounding,
        "action_kind": args.action_kind,
        "metadata": {
            "split": args.split,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
        },
    }
    torch.save(payload, path)
    print(f"Checkpoint saved to {path}")


def main() -> None:
    args = parse_args()
    dataset, action_values, rides = collect_samples(
        encoded_root=args.encoded_root,
        actions_root=args.actions_root,
        split=args.split,
        action_kind=args.action_kind,
        ride_ids=args.ride_ids,
        pool_hw=args.pool_size,
        rounding=args.rounding,
    )

    print(
        json.dumps(
            {
                "num_samples": len(dataset),
                "num_classes": int(action_values.shape[0]),
                "feature_dim": int(dataset.features.shape[1]),
                "rides": sorted(set(rides)),
            },
            indent=2,
        )
    )

    model, accuracy = train_model(dataset, action_values, args)

    if args.checkpoint_path:
        save_checkpoint(args.checkpoint_path, model, action_values, args)

    print(f"Training complete. Final accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
