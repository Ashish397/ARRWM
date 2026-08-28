import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from model.tied_residual_vjp import StructuredResidualVJP


VAE_PATH = Path(__file__).resolve().parents[1] / "wan/modules/vae.py"
SPEC = importlib.util.spec_from_file_location("_test_tied_wan_vae", VAE_PATH)
assert SPEC is not None and SPEC.loader is not None
VAE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VAE)
ResidualBlock = VAE.ResidualBlock


def test_structured_full_rank_matches_autograd_and_is_linear():
    torch.manual_seed(17)
    block = ResidualBlock(4, 4).eval().requires_grad_(False)
    state = torch.randn(2, 4, 3, 6, 7)
    first = torch.randn_like(state)
    second = torch.randn_like(state)
    exact_input = state.clone().requires_grad_(True)
    exact = torch.autograd.grad(
        block(exact_input), exact_input, grad_outputs=first,
    )[0]
    operator = StructuredResidualVJP(block)
    with torch.no_grad():
        predicted = operator(state, first)
        combined = operator(state, 0.3 * first - 0.8 * second)
        separate = 0.3 * operator(state, first) - 0.8 * operator(state, second)
    torch.testing.assert_close(predicted, exact, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(combined, separate, rtol=3e-5, atol=3e-5)
    assert float(F.cosine_similarity(predicted.flatten(1), exact.flatten(1)).min()) > 0.99999


def test_channel_changing_shortcut_matches_autograd():
    torch.manual_seed(19)
    block = ResidualBlock(3, 5).eval().requires_grad_(False)
    state = torch.randn(1, 3, 2, 5, 6)
    cotangent = torch.randn(1, 5, 2, 5, 6)
    exact_input = state.clone().requires_grad_(True)
    exact = torch.autograd.grad(
        block(exact_input), exact_input, grad_outputs=cotangent,
    )[0]
    with torch.no_grad():
        predicted = StructuredResidualVJP(block)(state, cotangent)
    torch.testing.assert_close(predicted, exact, rtol=3e-5, atol=3e-5)


def test_tied_low_rank_operator_remains_linear():
    torch.manual_seed(23)
    block = ResidualBlock(5, 5).eval().requires_grad_(False)
    state = torch.randn(1, 5, 2, 5, 6)
    first = torch.randn_like(state)
    second = torch.randn_like(state)
    operator = StructuredResidualVJP(block, rank=3)
    with torch.no_grad():
        combined = operator(state, 0.2 * first + 0.6 * second)
        separate = 0.2 * operator(state, first) + 0.6 * operator(state, second)
    torch.testing.assert_close(combined, separate, rtol=5e-5, atol=5e-5)
