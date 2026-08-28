import inspect
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis/gan_tuning/surrogate_rl_controller_benchmark.py"
SPEC = importlib.util.spec_from_file_location("_surrogate_rl_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from _surrogate_rl_benchmark import (  # noqa: E402
    ACTION_SPECS,
    OBSERVATION_NAMES,
    BankData,
    LinearUCBBandit,
    ObservationMemory,
    _model_metrics,
    _new_student,
    _teacher_drift,
    build_observation,
    fit_interval,
    nested_split,
    run_trajectory,
    validate_global_isolation,
    validation_reward,
)


ROLE_RINGS = {
    0: {"teacher": (0, 1), "surrogate_train": (2, 3, 4),
        "surrogate_test": (5, 6, 7)},
    1: {"teacher": (2, 3), "surrogate_train": (4, 5, 6),
        "surrogate_test": (7, 0, 1)},
    2: {"teacher": (4, 5), "surrogate_train": (6, 7, 0),
        "surrogate_test": (1, 2, 3)},
}


def _rank_rides(rank):
    return [f"rank{rank}_ride{i}" for i in range(3)]


def _protocol_bank(rotation, *, steps=(0,), spatial=4):
    ring = ROLE_RINGS[int(rotation)]
    samples = []
    for role in ("surrogate_train", "surrogate_test"):
        for rank in ring[role]:
            for ride in _rank_rides(rank):
                for step in steps:
                    samples.append((role, rank, ride, step))
    n = len(samples)
    generator = torch.Generator().manual_seed(100 + int(rotation))
    latent = torch.randn(n, 1, 16, spatial, spatial, generator=generator)
    target = torch.randn(n, 1, 16, spatial, spatial, generator=generator)
    dims = tuple(range(1, target.dim()))
    target = target / target.pow(2).mean(
        dim=dims, keepdim=True,
    ).sqrt().clamp_min(1.0e-12)
    roles = np.asarray([item[0] for item in samples])
    ranks = np.asarray([item[1] for item in samples], dtype=np.int64)
    rides = np.asarray([item[2] for item in samples])
    step_values = np.asarray([item[3] for item in samples], dtype=np.int64)
    metadata = {
        "source": "vgg", "rotation": int(rotation),
        "split": {
            "rank_roles": {key: list(value) for key, value in ring.items()},
            "teacher": [
                ride for rank in ring["teacher"] for ride in _rank_rides(rank)
            ],
        },
    }
    return BankData(
        path=Path(f"synthetic_r{rotation}.npz"), rotation=int(rotation),
        latent=latent, target=target,
        teacher_rms=np.linspace(0.001, 0.002, n), rides=rides, roles=roles,
        ranks=ranks, origins=np.zeros((n, 2), dtype=np.int64),
        steps=step_values, crop_indices=np.zeros(n, dtype=np.int64),
        source_units=np.asarray([f"unit{i}" for i in range(n)]),
        metadata=metadata,
    )


def test_nested_ride_isolation_for_every_controller_fold():
    banks = [_protocol_bank(rotation) for rotation in range(3)]
    audit = validate_global_isolation(banks)
    assert audit["controller_fit_is_fold_specific"] is True
    assert audit["shared_policy_design_and_hyperparameters"] is True
    for bank in banks:
        split = nested_split(bank)
        roles = [
            set(split.teacher_rides), set(split.train_rides),
            set(split.validation_rides), set(split.test_rides),
        ]
        assert all(
            not left & right
            for index, left in enumerate(roles) for right in roles[index + 1:]
        )


def test_observation_and_reward_interfaces_are_causal_and_test_free():
    observation_parameters = set(inspect.signature(build_observation).parameters)
    assert not observation_parameters & {
        "validation", "test", "future", "reward", "next_teacher",
    }
    reward_parameters = set(inspect.signature(validation_reward).parameters)
    assert not reward_parameters & {"train", "test", "fitted_batch"}
    kwargs = dict(
        step_index=1, n_steps=9,
        pre_train={"q1": 0.1, "median": 0.2, "mse": 0.9},
        teacher_rms=0.004, teacher_drift=0.7,
        memory=ObservationMemory(previous_action=2),
    )
    first = build_observation(**kwargs)
    second = build_observation(**kwargs)
    assert first.shape == (len(OBSERVATION_NAMES),)
    assert np.array_equal(first, second)


def test_final_test_target_mutation_cannot_change_train_observation_inputs():
    bank = _protocol_bank(0, steps=(0, 5))
    split = nested_split(bank)
    train_now = bank.ids(step=5, ranks=split.train_ranks)
    train_previous = bank.ids(step=0, ranks=split.train_ranks)
    before = _teacher_drift(bank, train_now, train_previous)
    mutated = bank.target.clone()
    test_ids = bank.ids(step=5, ranks=split.test_ranks)
    mutated[test_ids].mul_(-1000.0)
    bank.target = mutated
    after = _teacher_drift(bank, train_now, train_previous)
    assert before == after


def test_final_test_target_mutation_cannot_steer_frozen_bandit():
    bank = _protocol_bank(0, steps=(0, 5))
    policy = LinearUCBBandit(alpha=0.25, ridge=1.0)
    observation = np.linspace(-0.2, 0.2, len(OBSERVATION_NAMES))
    for action, reward in enumerate((0.1, 0.2, -0.1, 0.0)):
        policy.update(observation, action, reward)
    kwargs = dict(
        method="bandit", student_seed=456, policy=policy,
        substeps=1, batch_size=3, width=8, blocks=1, max_steps=2,
        device=torch.device("cpu"),
    )
    original = run_trajectory(bank, **kwargs)
    split = nested_split(bank)
    test_ids = np.flatnonzero(np.isin(bank.ranks, split.test_ranks))
    bank.target[test_ids] = -bank.target[test_ids]
    mutated = run_trajectory(bank, **kwargs)
    assert [item["action"] for item in original["records"]] == [
        item["action"] for item in mutated["records"]
    ]
    assert [item["observation"] for item in original["records"]] == [
        item["observation"] for item in mutated["records"]
    ]
    assert original["final_current_q1"] != mutated["final_current_q1"]


def test_deterministic_fit_seed_and_compute_budget_accounting():
    bank = _protocol_bank(0)
    split = nested_split(bank)
    ids = bank.ids(step=0, ranks=split.train_ranks)
    for action in range(len(ACTION_SPECS)):
        model_a, opt_a = _new_student(
            seed=123, width=8, blocks=1, device=torch.device("cpu")
        )
        model_b, opt_b = _new_student(
            seed=123, width=8, blocks=1, device=torch.device("cpu")
        )
        fit_a = fit_interval(
            model_a, opt_a, bank, ids, action=action, substeps=2,
            batch_size=3, seed=991, step_index=0, device=torch.device("cpu"),
        )
        fit_b = fit_interval(
            model_b, opt_b, bank, ids, action=action, substeps=2,
            batch_size=3, seed=991, step_index=0, device=torch.device("cpu"),
        )
        assert fit_a == fit_b
        assert fit_a["updates"] == 2
        assert all(
            torch.equal(left, right)
            for left, right in zip(model_a.state_dict().values(),
                                   model_b.state_dict().values())
        )


def test_fixed_policy_is_equivalent_to_direct_supervised_recipe():
    bank = _protocol_bank(0)
    split = nested_split(bank)
    trajectory = run_trajectory(
        bank, method="fixed", student_seed=321, policy=None,
        substeps=2, batch_size=3, width=8, blocks=1, max_steps=1,
        device=torch.device("cpu"),
    )
    assert trajectory["selected_student_updates"] == 2
    assert trajectory["search_student_updates"] == 0
    assert [item["action"] for item in trajectory["records"]] == [0]

    model, optimizer = _new_student(
        seed=321, width=8, blocks=1, device=torch.device("cpu")
    )
    ids = bank.ids(step=0, ranks=split.train_ranks)
    fit_interval(
        model, optimizer, bank, ids, action=0, substeps=2, batch_size=3,
        seed=321, step_index=0, device=torch.device("cpu"),
    )
    test_ids = bank.ids(step=0, ranks=split.test_ranks)
    reference = _model_metrics(
        model, bank, test_ids, device=torch.device("cpu")
    )
    assert trajectory["final_current_q1"] == reference["q1"]


def test_bandit_policy_roundtrip_is_deterministic():
    policy = LinearUCBBandit(alpha=0.25, ridge=2.0)
    observation = np.linspace(-0.5, 0.5, len(OBSERVATION_NAMES))
    for reward in (0.1, -0.2, 0.3, 0.0):
        action = policy.choose(observation, explore=True)
        policy.update(observation, action, reward)
    restored = LinearUCBBandit.from_dict(json.loads(json.dumps(policy.to_dict())))
    assert restored.choose(observation, explore=False) == policy.choose(
        observation, explore=False
    )
    assert restored.counts == policy.counts


def test_fold_policy_uses_nested_train_and_validation_ranks_only():
    bank = _protocol_bank(0)
    split = nested_split(bank)
    assert split.train_ranks == (2, 3)
    assert split.validation_ranks == (4,)
    assert set(split.train_rides).isdisjoint(split.validation_rides)
    assert set(split.validation_rides).isdisjoint(split.test_rides)
    assert all(item.startswith(("rank2_", "rank3_"))
               for item in split.train_rides)
    assert all(item.startswith("rank4_") for item in split.validation_rides)
