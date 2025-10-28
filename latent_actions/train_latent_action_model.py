#!/usr/bin/env python3
"""Train an ensemble of latent action classifiers that infer actions from states.

The script pairs encoded frame tensors (s_t, s_{t+1}) with either input or output
actions and trains an ensemble of simple heads (stacked, convolved, concatenated,
subtracted views) to predict the corresponding action class. It expects the
encoded tensors and action CSVs to share the same directory layout and can be
configured via `latent_actions/latent_training.yaml`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TransitionDataset(Dataset):
    def __init__(self, variant_names: Sequence[str], feature_tensors: Dict[str, torch.Tensor], labels: torch.Tensor, rides: List[str]) -> None:
        self.variant_names = list(variant_names)
        self._features = [feature_tensors[name] for name in self.variant_names]
        self._labels = labels
        self._rides = rides

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return [feat[idx] for feat in self._features], self._labels[idx]

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @property
    def rides(self) -> List[str]:
        return list(self._rides)

    def feature_tensors(self) -> Dict[str, torch.Tensor]:
        return {name: feat for name, feat in zip(self.variant_names, self._features)}


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


class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        conv_hidden = max(32, min(256, in_channels * 2))
        conv_out = max(64, conv_hidden)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_hidden, conv_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = MLPClassifier(conv_out, hidden_dims, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        flattened = features.flatten(start_dim=1)
        return self.classifier(flattened)


class EnsembleClassifier(nn.Module):
    def __init__(
        self,
        variant_names: Sequence[str],
        submodules: Dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.variant_names = list(variant_names)
        self.submodules = nn.ModuleDict(submodules)

    def forward(self, *features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        component_logits: Dict[str, torch.Tensor] = {}
        logits_list: List[torch.Tensor] = []
        for name, feature in zip(self.variant_names, features):
            logits = self.submodules[name](feature)
            component_logits[name] = logits
            logits_list.append(logits)
        stacked_logits = torch.stack(logits_list, dim=0)
        return stacked_logits.mean(dim=0), component_logits


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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("latent_actions/latent_training.yaml"),
        help="Optional configuration file to control ensemble components (JSON-formatted YAML).",
    )
    return parser.parse_args()


def infer_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_config(path: Path) -> Dict[str, object]:
    if not path:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Configuration file {path} must contain JSON-compatible YAML content.") from exc


def resolve_ensemble_flags(config: Dict[str, object]) -> Dict[str, bool]:
    defaults = {
        "stacked": True,
        "convolved": True,
        "concatenated": True,
        "subtracted": True,
    }
    ensemble_cfg = config.get("ensemble") if isinstance(config, dict) else None
    if isinstance(ensemble_cfg, dict):
        overrides = {key: bool(value) for key, value in ensemble_cfg.items() if key in defaults}
        defaults.update(overrides)
    if not any(defaults.values()):
        raise ValueError("Ensemble configuration disables all models; enable at least one.")
    return defaults



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


def pool_state(state: torch.Tensor, pool_hw: int) -> torch.Tensor:
    if state.dim() == 3:
        return F.adaptive_avg_pool2d(state.unsqueeze(0), (pool_hw, pool_hw)).squeeze(0)
    if state.dim() == 2:
        return F.adaptive_avg_pool2d(state.unsqueeze(0).unsqueeze(0), (pool_hw, pool_hw)).squeeze(0).squeeze(0)
    raise ValueError(f"Unsupported state tensor with shape {tuple(state.shape)}")


def build_variant_feature(variant: str, state_t: torch.Tensor, state_tp1: torch.Tensor) -> torch.Tensor:
    if variant == "stacked":
        return torch.cat([state_t, state_tp1], dim=0).flatten()
    if variant == "concatenated":
        return torch.cat([state_t.flatten(), state_tp1.flatten()], dim=0)
    if variant == "subtracted":
        return (state_tp1 - state_t).flatten()
    if variant == "convolved":
        return torch.cat([state_t, state_tp1], dim=0)
    raise KeyError(f"Unknown variant '{variant}'")


def collect_samples(
    encoded_root: Path,
    actions_root: Path,
    split: str,
    action_kind: str,
    ride_ids: Optional[Sequence[str]],
    active_variants: Sequence[str],
    pool_hw: int,
    rounding: int,
) -> Tuple[TransitionDataset, torch.Tensor, List[str]]:
    action_csvs = find_action_files(actions_root, split, action_kind, ride_ids)
    variant_features: Dict[str, List[torch.Tensor]] = {name: [] for name in active_variants}
    labels: List[int] = []
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
        start_label_count = len(labels)
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
                state_t = frames[idx].float()
                state_tp1 = frames[idx + 1].float()
                pooled_t = pool_state(state_t, pool_hw)
                pooled_tp1 = pool_state(state_tp1, pool_hw)
                for variant in active_variants:
                    feature_tensor = build_variant_feature(variant, pooled_t, pooled_tp1)
                    variant_features[variant].append(feature_tensor)
                action_vec = action_values[action_cursor + idx]
                class_id = assign_action_class(action_vec, action_lookup, action_prototypes, rounding)
                labels.append(class_id)
                rides_used.append(ride_name)
            action_cursor += max_pairs

        if len(labels) == start_label_count:
            raise ValueError(f"No frames loaded for ride '{ride_name}'")

    if not labels:
        raise RuntimeError("No samples constructed from the provided data.")
    if not action_prototypes:
        raise RuntimeError("Failed to derive any unique actions from the provided CSV files.")

    feature_tensors = {}
    for variant_name, feature_list in variant_features.items():
        if not feature_list:
            continue
        stacked = torch.stack(feature_list)
        feature_tensors[variant_name] = stacked
    missing_variants = [name for name in active_variants if name not in feature_tensors]
    if missing_variants:
        raise RuntimeError(f"No features were generated for variants: {missing_variants}")
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TransitionDataset(active_variants, feature_tensors, labels_tensor, rides_used)
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
) -> Tuple[EnsembleClassifier, float]:
    device = infer_device(args.device)
    feature_map = dataset.feature_tensors()
    submodules: Dict[str, nn.Module] = {}
    for name in dataset.variant_names:
        tensor = feature_map[name]
        if tensor.dim() == 2:
            input_dim = tensor.shape[1]
            submodules[name] = MLPClassifier(input_dim, args.hidden_dims, action_values.shape[0], dropout=args.dropout)
        elif tensor.dim() == 4:
            _, in_channels, _, _ = tensor.shape
            submodules[name] = ConvClassifier(in_channels, action_values.shape[0], args.hidden_dims, dropout=args.dropout)
        else:
            raise ValueError(f"Unsupported feature tensor rank {tensor.dim()} for variant '{name}'")

    ensemble = EnsembleClassifier(dataset.variant_names, submodules).to(device)
    multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1
    if multi_gpu:
        ensemble = nn.DataParallel(ensemble)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    target_accuracy = args.min_accuracy
    eval_features = [feature_map[name].to(device).float() for name in dataset.variant_names]
    eval_labels = dataset.labels.to(device)
    history_loss: List[float] = []
    history_acc: List[float] = []

    for epoch in range(1, args.max_epochs + 1):
        ensemble.train()
        for batch_features, batch_labels in loader:
            batch_tensors = [tensor.to(device).float() for tensor in batch_features]
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, _ = ensemble(*batch_tensors)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            ensemble.eval()
            logits, _ = ensemble(*eval_features)
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

    trained_model = ensemble.module if isinstance(ensemble, nn.DataParallel) else ensemble
    return trained_model, final_accuracy


def save_checkpoint(
    path: Path,
    model: nn.Module,
    action_values: torch.Tensor,
    args: argparse.Namespace,
    variant_names: Sequence[str],
    ensemble_flags: Dict[str, bool],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "action_values": action_values,
        "hidden_dims": list(args.hidden_dims),
        "pool_size": args.pool_size,
        "rounding": args.rounding,
        "action_kind": args.action_kind,
        "variant_names": list(variant_names),
        "ensemble_flags": ensemble_flags,
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
    config = load_config(args.config)
    ensemble_flags = resolve_ensemble_flags(config)
    active_variants = [name for name, enabled in ensemble_flags.items() if enabled]
    dataset, action_values, rides = collect_samples(
        encoded_root=args.encoded_root,
        actions_root=args.actions_root,
        split=args.split,
        action_kind=args.action_kind,
        ride_ids=args.ride_ids,
        active_variants=active_variants,
        pool_hw=args.pool_size,
        rounding=args.rounding,
    )

    feature_shapes = {
        name: list(dataset.feature_tensors()[name].shape[1:])
        for name in dataset.variant_names
    }
    summary = {
        "num_samples": len(dataset),
        "num_classes": int(action_values.shape[0]),
        "variants": dataset.variant_names,
        "feature_shapes": feature_shapes,
        "ensemble_flags": ensemble_flags,
        "rides": sorted(set(rides)),
    }
    print(json.dumps(summary, indent=2))

    model, accuracy = train_model(dataset, action_values, args)

    if args.checkpoint_path:
        save_checkpoint(args.checkpoint_path, model, action_values, args, dataset.variant_names, ensemble_flags)

    print(f"Training complete. Final accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
