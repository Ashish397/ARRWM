import torch
import torch.nn.functional as F

from model.ladd_disc import LADDDiscriminator


class _Projector:
    def __call__(self, x_noisy, **_kwargs):
        b, f, _c, h, w = x_noisy.shape
        spatial = x_noisy.mean(dim=2).reshape(b * f, 1, h, w)
        tokens = F.avg_pool2d(spatial, 2).reshape(b, f * (h // 2) * (w // 2), 1)
        return {0: tokens.expand(-1, -1, 8)}


def _build(*, scalar_output, freeze_projector_mixing=False):
    return LADDDiscriminator(
        projector=_Projector(),
        block_indices=[0],
        dim_teacher=8,
        dim_proj=8,
        use_csm=False,
        cmap_dim=0,
        patch_size=(1, 2, 2),
        scalar_output=scalar_output,
        freeze_projector_mixing=freeze_projector_mixing,
    )


def test_scalar_output_is_one_differentiable_logit_per_sample():
    disc = _build(scalar_output=True)
    x = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
    logits = disc(
        x,
        timestep=torch.zeros(2, 2, dtype=torch.long),
        prompt_embeds=torch.zeros(2, 1, 8),
    )

    assert logits.shape == (2, 1)
    logits.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_frozen_projector_mixing_leaves_heads_trainable():
    disc = _build(scalar_output=True, freeze_projector_mixing=True)

    assert not any(p.requires_grad for p in disc.ccm.parameters())
    assert any(p.requires_grad for p in disc.heads.parameters())
