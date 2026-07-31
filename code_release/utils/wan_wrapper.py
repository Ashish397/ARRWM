# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import os
import types
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Optional, Union
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel

# Load default config to get wan_model_path
default_config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
_default_config = OmegaConf.load(default_config_path)
_default_wan_model_path = _default_config.get("wan_model_path", os.environ.get("DATA_ROOT", ""))
# Ensure path ends with a slash
if not _default_wan_model_path.endswith("/"):
    _default_wan_model_path = _default_wan_model_path + "/"


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_root: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name or "Wan2.1-T2V-1.3B"
        self.model_root = Path(model_root) if model_root is not None else Path(_default_wan_model_path) / self.model_name
        weights_path = self.model_root / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer_path = self.model_root / "google" / "umt5-xxl"

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(
                str(weights_path),
                map_location='cpu',
                weights_only=False,
            )
        )

        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path),
            seq_len=512,
            clean='whitespace',
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model_name = model_name or "Wan2.1-T2V-1.3B"
        self.model_root = Path(model_root) if model_root is not None else Path(_default_wan_model_path) / self.model_name
        vae_checkpoint = self.model_root / "Wan2.1_VAE.pth"

        # init model
        self.model = _video_vae(
            pretrained_path=str(vae_checkpoint),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        use_cache: bool = False,
        seed_first: bool = False,
    ) -> torch.Tensor:
        """Decode latents ``[B, F_lat, C, H, W]`` -> pixels ``[B, F_pix, 3, H, W]``.

        ``use_cache`` routes to ``cached_decode``, which does NOT clear the WAN
        VAE temporal-conv ``feat_map`` between calls — correct ONLY for genuine
        sequential continuation of the SAME video (the cache carries the real
        left-context). For an INDEPENDENT clip it is a bug: the decoder's first
        frame is convolved with whatever the previous (unrelated) decode left in
        the cache, so a ghost of another frame bleeds into frame 0.

        ``seed_first`` is the correct mode for any standalone render/grad decode.
        It SELF-SEEDS: prepend a replica of the clip's OWN first latent frame as
        a dummy, decode through the cache-clearing ``decode`` (so no stale
        cross-clip context survives), and slice the dummy's single special-first
        pixel frame off the front. The dummy is consumed as the WAN VAE
        "special first" (1 pixel frame) and seeds the temporal cache so the real
        first frame has a FAITHFUL (self) predecessor — no cross-clip ghost AND
        no plain-decode init-frame brightness anomaly. ``seed_first`` overrides
        ``use_cache``.
        """
        if seed_first:
            # Prepend a replica of the clip's own first latent frame (frame dim).
            latent = torch.cat([latent[:, 0:1], latent], dim=1)
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache and not seed_first:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache and not seed_first:
            decode_function = self.model.cached_decode
        else:
            # ``decode`` brackets the loop with ``clear_cache()`` (start + end),
            # so each clip is isolated; under ``seed_first`` the prepended dummy
            # seeds the cache within this single call.
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        if seed_first:
            # Drop the dummy's single special-first pixel frame.
            output = output[:, 1:]
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()
        self.model_name = model_name
        self.model_root = Path("wan_models") / self.model_name

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"{_default_wan_model_path}{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"{_default_wan_model_path}{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]
        self._base_seq_len = self.seq_len


        self.post_init()

    def adjust_seq_len_for_action_tokens(self, num_frames: int = 21, action_per_frame: int = 1):
        """Increase seq_len capacity to accommodate per-frame action tokens."""
        self.seq_len = self._base_seq_len + num_frames * action_per_frame

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _unwrapped_model(self):
        """Return the underlying nn.Module after stripping a DDP wrap.

        After ``trainer.causal_action_forcing_train`` re-assigns
        ``model.fake_score.model = DDP(...)``, this wrapper's ``self.model``
        is the DDP instance, which doesn't proxy arbitrary attribute
        access (e.g. ``head_alt`` lives on the wrapped ``module``, not
        on the DDP itself). Use this helper anywhere the wrapper code
        needs to read attributes of the actual WAN model.
        """
        m = self.model
        try:
            from torch.nn.parallel import DistributedDataParallel as _DDP
            if isinstance(m, _DDP):
                m = m.module
        except Exception:
            pass
        return m


        # self.has_cls_branch = True

    def adding_state_token_branch(
        self,
        n_chunks: int = 7,
        z_out_dim: int = 2,
        dim: int = 2048,
        num_frame_per_block: int = 3,
    ) -> None:
        """Add per-frame state tokens that evolve inside the transformer.

        Creates learned initial state embeddings (one per *frame*) that are
        inserted per-frame into the sequence alongside visual and action
        tokens.  After the transformer, state hidden states are pooled
        per-chunk and mapped to the teacher latent via a linear readout.

        Must be called before DDP wrapping and after action token setup.
        """
        n_frames = n_chunks * num_frame_per_block
        self._state_token_init = nn.Parameter(
            torch.randn(n_frames, dim) * 0.02,
        )
        self._state_readout = nn.Linear(dim, z_out_dim)
        nn.init.normal_(self._state_readout.weight, std=1e-3)
        nn.init.zeros_(self._state_readout.bias)

        self._state_n_chunks = n_chunks
        self._state_z_out_dim = z_out_dim
        self._state_num_frame_per_block = num_frame_per_block

        # Unwrap PeftModel (if LoRA has been applied) so attributes land on
        # the actual CausalWanModel that reads them inside _forward_train.
        base_model = self.model
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        base_model.state_tokens_per_frame = 1
        base_model.action_tokens_per_frame += 1
        self.seq_len += n_frames

    def _build_state_tokens(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Expand per-frame learned state inits to [B, F, dim]."""
        st = self._state_token_init.unsqueeze(0).expand(batch_size, -1, -1)
        return st.to(dtype=dtype, device=device)

    def adding_state_probe_branch(
        self,
        n_chunks: int = 7,
        z_out_dim: int = 2,
        dim: int = 2048,
        probe_dim: int = 256,
        num_heads: int = 8,
        n_taps: int = 6,
        num_frame_per_block: int = 3,
    ) -> None:
        """Add cross-attention state probes that tap transformer features at
        multiple depths instead of injecting state tokens into the sequence.

        Must be called before DDP wrapping and after LoRA application.
        """
        from wan.modules.causal_model import StateProbeModule

        base_model = self.model
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()

        a_per_f = int(getattr(base_model, 'action_tokens_per_frame', 0))
        self._state_probe = StateProbeModule(
            n_chunks=n_chunks,
            model_dim=dim,
            probe_dim=probe_dim,
            z_out_dim=z_out_dim,
            num_heads=num_heads,
            n_taps=n_taps,
            num_frame_per_block=num_frame_per_block,
            action_tokens_per_frame=a_per_f,
        )
        self._state_n_chunks = n_chunks
        self._state_z_out_dim = z_out_dim

        n_blocks = len(base_model.blocks)
        tap_indices = [int(round(i * (n_blocks - 1) / (n_taps - 1))) for i in range(n_taps)]
        base_model._state_probe_tap_set = set(tap_indices)
        base_model._state_probe_tap_indices = tap_indices

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        if getattr(self, "_action_patch_applied", False):
            action_mod_kwargs = {}
            action_modulation = conditional_dict.get("_action_modulation", None)
            if action_modulation is not None:
                action_mod_kwargs["action_modulation"] = action_modulation
            am_clean = conditional_dict.get("_action_modulation_clean", None)
            if am_clean is not None:
                action_mod_kwargs["action_modulation_clean"] = am_clean
            at = conditional_dict.get("_action_tokens", None)
            if at is not None:
                action_mod_kwargs["action_tokens"] = at
            at_clean = conditional_dict.get("_action_tokens_clean", None)
            if at_clean is not None:
                action_mod_kwargs["action_tokens_clean"] = at_clean
        else:
            action_mod_kwargs = {}

        # [B, F] -> [B]
        # if self.uniform_timestep:
        #     input_timestep = timestep[:, 0]
        # else:
        #     input_timestep = timestep
        input_timestep = timestep

        has_state = getattr(self, "_state_token_init", None) is not None
        has_probe = getattr(self, "_state_probe", None) is not None
        state_hidden = None
        tapped_features = None

        # Build state tokens once. Cached inference only supports the noisy-side
        # tokens, while teacher-forcing also threads a clean-side copy.
        state_kwargs = {}
        if has_state:
            B = noisy_image_or_video.shape[0]
            st = self._build_state_tokens(B, noisy_image_or_video.dtype, noisy_image_or_video.device)
            if kv_cache is not None:
                model_for_shape = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
                patch_size = getattr(model_for_shape, "patch_size", (1, 2, 2))
                spatial_tokens_per_frame = (
                    noisy_image_or_video.shape[-2] * noisy_image_or_video.shape[-1]
                ) // (patch_size[1] * patch_size[2])
                tokens_per_frame = spatial_tokens_per_frame + int(
                    getattr(model_for_shape, "action_tokens_per_frame", 0)
                )
                frame_start = int(current_start or 0) // max(tokens_per_frame, 1)
                frame_end = frame_start + noisy_image_or_video.shape[1]
                st = st[:, frame_start:frame_end].contiguous()
            state_kwargs["state_tokens"] = st
            if clean_x is not None and kv_cache is None:
                state_kwargs["state_tokens_clean"] = st

        # X0 prediction
        if kv_cache is not None:
            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                **action_mod_kwargs,
                **state_kwargs,
            )
            # Return contract (mirrors _forward_inference):
            #   plain tensor                      -> flow_pred only
            #   (tensor, state_hidden)            -> action-token state path
            #   (tensor, tapped_infer)            -> state-probe taps only
            #   (tensor, state_hidden, tapped)    -> both
            if isinstance(model_out, tuple):
                flow_pred = model_out[0].permute(0, 2, 1, 3, 4)
                if len(model_out) == 3:
                    state_hidden = model_out[1]
                    tapped_features = model_out[2]
                else:
                    aux = model_out[1]
                    if isinstance(aux, list):
                        tapped_features = aux
                    else:
                        state_hidden = aux
            else:
                flow_pred = model_out.permute(0, 2, 1, 3, 4)
        elif clean_x is not None:
            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                clean_x=clean_x.permute(0, 2, 1, 3, 4),
                aug_t=aug_t,
                **action_mod_kwargs,
                **state_kwargs,
            )
            if isinstance(model_out, tuple):
                flow_pred = model_out[0].permute(0, 2, 1, 3, 4)
                aux = model_out[1]
                if isinstance(aux, list):
                    tapped_features = aux
                else:
                    state_hidden = aux
            else:
                flow_pred = model_out.permute(0, 2, 1, 3, 4)
        else:
            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                **action_mod_kwargs,
                **state_kwargs,
            )
            if isinstance(model_out, tuple):
                flow_pred = model_out[0].permute(0, 2, 1, 3, 4)
                aux = model_out[1]
                if isinstance(aux, list):
                    tapped_features = aux
                else:
                    state_hidden = aux
            else:
                flow_pred = model_out.permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        # Cross-attention probe readout
        if has_probe and tapped_features is not None:
            noisy_start = tapped_features[0].shape[1] // 2 if clean_x is not None else 0
            # Compute tokens-per-frame from the noisy-side sequence length
            num_frames = noisy_image_or_video.shape[1]
            noisy_seq = tapped_features[0].shape[1] - noisy_start
            frame_seqlen = noisy_seq // num_frames
            # The probe's `n_chunks * num_frame_per_block` window must
            # exactly equal `num_frames` — otherwise the chunk-wise reshape
            # inside `StateProbeModule.forward` asserts/shape-errors.
            # In the rolling-staircase pipeline only the live-window forward
            # (12 frames = 4 slots x 3 fpb) satisfies this; priming forwards
            # (3 frames each) and commit forwards (3 frames each) do not.
            # Silently skip the probe for those and fall through to the
            # no-probe return — the taps are collected but discarded.
            probe_n_chunks = int(getattr(self, "_state_n_chunks", 0))
            # ``self._state_probe`` may be DDP-wrapped (when the trainer
            # builds a separate optimizer for state_probe and wraps it
            # for grad sync). DDP doesn't proxy arbitrary user attrs
            # through ``__getattr__``, so ``getattr(ddp_wrap, "num_frame_
            # per_block", 0)`` returns 0 unless the attribute is
            # explicitly set on the wrap. Unwrap before reading to make
            # this robust regardless of trainer-side fixups.
            probe_module = self._state_probe
            try:
                from torch.nn.parallel import DistributedDataParallel as _DDP
                if isinstance(probe_module, _DDP):
                    probe_module = probe_module.module
            except Exception:
                pass
            probe_fpb = int(getattr(
                probe_module, "num_frame_per_block", 0,
            ) or 0)
            expected_frames = probe_n_chunks * probe_fpb
            # Dual-view rear camera teacher path feeds 2x frames (forward+reverse). The
            # probe runs on the FORWARD half only: StateProbeModule slices the
            # first n_chunks*chunk_tokens tokens and forward frames are first, and
            # frame_seqlen above is computed per-frame, so the reshape stays valid.
            # Accept exact (single-view) OR 2x (dual-view) the probe window.
            if expected_frames > 0 and num_frames in (expected_frames, 2 * expected_frames):
                # Memory-vs-grad-coverage trade: backproping through the
                # taps into the underlying DiT requires keeping the full
                # forward graph alive between the model forward and the
                # state_probe loss backward. With LoRA-active real_score
                # + 6 taps × ~28k tokens × 1536 dim, that pushes the
                # 95 GB H100 budget over the edge during the gen-step
                # (real_score forward graph + DMD aux loss graph + GAN
                # graph all coexist). Detach the taps so state_probe
                # backward only reaches the probe's own params; the
                # upstream model is supervised by other losses
                # (FlowPredLoss in the aux pass, action_critic
                # z-guidance via _x0). Gated by
                # ``self._state_probe_detach_taps`` so the trainer can
                # opt into full-graph mode if memory headroom permits.
                taps_for_probe = tapped_features
                if bool(getattr(self, "_state_probe_detach_taps", True)):
                    taps_for_probe = [t.detach() for t in tapped_features]
                state_preds, probe_hidden = self._state_probe(
                    taps_for_probe, noisy_start, frame_seqlen,
                )
                return flow_pred, pred_x0, state_preds.float(), probe_hidden

        # State-token readout: pool per chunk and map to (PC0, PC1)
        if has_state and state_hidden is not None:
            fpb = self._state_num_frame_per_block
            n_c = self._state_n_chunks
            B = state_hidden.shape[0]
            actual_frames = state_hidden.shape[1]
            if kv_cache is not None and actual_frames < n_c * fpb:
                actual_chunks = max(1, actual_frames // fpb)
                used = actual_chunks * fpb
                pooled = state_hidden[:, :used].reshape(B, actual_chunks, fpb, -1).mean(dim=2)
            else:
                pooled = state_hidden[:, :n_c * fpb].reshape(B, n_c, fpb, -1).mean(dim=2)
            readout_dtype = next(self._state_readout.parameters()).dtype
            state_preds = self._state_readout(pooled.to(readout_dtype)).float()
            return flow_pred, pred_x0, state_preds, pooled

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
