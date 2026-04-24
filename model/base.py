# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from typing import Tuple, Optional
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline, ActionSelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from model.action_model_patch import apply_action_patches, apply_action_patches_critic
from model.action_modulation import ActionModulationProjection, ActionTokenProjection

from utils.debug_option import DEBUG

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        cfg_flag = getattr(args, "action_patch_enabled", None)
        self._action_patch_enabled = bool(cfg_flag)
        object.__setattr__(self, "_action_projection_ref", None)
        object.__setattr__(self, "_action_token_projection_ref", None)
        self._action_dim: Optional[int] = None
        self.text_pre_encoded = bool(getattr(args, "text_pre_encoded", False))
        # The action apparatus is an indivisible unit of four streams (AdaLN
        # modulation + per-frame action tokens, on both the noisy and clean-x
        # windows). The v14 teacher, ODE student, and Phase-1 DMD all train
        # with ``action_conditioning_mode="both"`` (strictly enforced in
        # ``action-forcing/af_model/ode_regression.py:138``). Running with
        # only one of the two streams means feeding the DiT an input
        # convention it was never trained on — AdaLN-only silently kills
        # the per-frame-action-token input AND drops the frame's sequence
        # length from 1561 to 1560, both of which are out-of-distribution.
        # We refuse to proceed in a degraded mode.
        if self._action_patch_enabled:
            acm = str(getattr(args, "action_conditioning_mode", "both"))
            if acm != "both":
                raise RuntimeError(
                    f"action_conditioning_mode must be 'both' for action-"
                    f"aware training (got {acm!r}). The ODE student and v14 "
                    f"teacher were both trained with both AdaLN modulation "
                    f"AND per-frame action tokens active; running with only "
                    f"one stream feeds the DiT an untrained input convention."
                )
            self._action_conditioning_mode = acm
        else:
            self._action_conditioning_mode = None
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        base_model_name = getattr(args, "model_name", None)
        self.real_model_name = getattr(args, "real_name", base_model_name) or "Wan2.1-T2V-1.3B"
        self.fake_model_name = getattr(args, "fake_name", base_model_name) or "Wan2.1-T2V-1.3B"

        model_kwargs = self._to_dict(getattr(args, "model_kwargs", None))
        self.local_attn_size = model_kwargs.get("local_attn_size", -1)
        if "model_name" not in model_kwargs:
            model_kwargs["model_name"] = self.real_model_name
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        # Apply action patches before any distributed/FSDP wrapping so the inner
        # module gains the extended signature and hooks.
        if self._action_patch_enabled:
            apply_action_patches(self.generator)
            self._init_action_projection(args, device)
            if self.action_projection is None:
                raise RuntimeError(
                    "Action patches are enabled but action_projection failed to initialize."
                )
            self.action_projection.requires_grad_(True)

            # Stream B: per-frame action tokens. Matches the ODE trainer's
            # `ActionTokenProjection` + `adjust_seq_len_for_action_tokens`
            # pair. Without this, the generator's per-frame sequence length
            # stays at 1560 (pure spatial), and the DiT weights trained
            # with 1561 tokens/frame (spatial + 1 action token) are run
            # out-of-distribution.
            self._init_action_token_projection(args, device)
            if self.action_token_projection is None:
                raise RuntimeError(
                    "Action patches are enabled and action_conditioning_mode"
                    "='both' requires Stream B (action-token projection) to "
                    "be live, but ``action_token_projection`` failed to "
                    "initialize."
                )
            self.action_token_projection.requires_grad_(True)
            # Wire the wrapper to reserve seq-len for 1 action token / frame
            # and tell the inner CausalWanModel how many action tokens it
            # should split off per frame.
            num_training_frames = int(
                getattr(args, "num_training_frames", 21)
            )
            self.generator.model.action_tokens_per_frame = 1
            self.generator.adjust_seq_len_for_action_tokens(
                num_frames=num_training_frames, action_per_frame=1,
            )
            # Defensive: fail loud if the downstream pipeline ends up
            # running without a matching ``action_tokens_per_frame > 0``
            # on the inner DiT (means an upstream refactor silently
            # dropped Stream B).
            if int(getattr(self.generator.model, "action_tokens_per_frame", 0)) != 1:
                raise RuntimeError(
                    "Generator inner model has action_tokens_per_frame != 1 "
                    "after Stream B wiring. The DiT will run at seq-len "
                    "1560/frame instead of the 1561/frame it was trained on."
                )

        self.generator.model.requires_grad_(True)

        # ------------------------------------------------------------------
        # real_score + fake_score (bidirectional WanModel).
        #
        # Both scorers must see the same two-stream action conditioning
        # the generator was trained on:
        #   - Stream A (AdaLN modulation)   via conditional_dict["_action_modulation"]
        #   - Stream B (per-frame tokens)   via conditional_dict["_action_tokens"]
        #
        # Without the bidirectional action patch applied, the wrapper's
        # forward silently drops BOTH streams — which is what was
        # happening to ``real_score`` before this fix. Apply the patch to
        # both, and configure ``action_tokens_per_frame = 1`` so the
        # patched bidirectional _forward interleaves per-frame tokens at
        # the trained 1561-tokens-per-frame layout.
        #
        # ``adjust_seq_len_for_action_tokens`` bumps the wrapper's
        # ``seq_len`` upper bound to accommodate the extra per-frame
        # token. For the scorers, seq_len MUST equal the actual
        # ``num_frames * spatial_seqlen (+ action_tokens)`` of the input
        # we feed them — NOT a max — because the bidirectional block's
        # internal ``unflatten(dim=1, sizes=(num_frames, frame_seqlen))``
        # computes ``frame_seqlen = x.shape[1] // e.shape[1]``. If we
        # over-pad (e.g. seq_len=32781 but actually pass 12 frames),
        # the integer division yields a garbage ``frame_seqlen`` and
        # the unflatten asserts.
        #
        # In rolling-staircase DMD the scorers are called with a 4-slot
        # batched window of exactly ``4 * num_frame_per_block`` frames
        # (= 3 context + 1 target per slot, concatenated), so we size
        # the scorers' ``_base_seq_len`` to that.
        # ------------------------------------------------------------------
        npb = int(getattr(args, "num_frame_per_block", 3))
        # Real-score window: (real_k GT + 1 prev + 1 live) * npb frames.
        # real_k defaults to 2 → 4 * npb (matches current 4-chunk window).
        # Bumping to 3 or 4 widens real_score's GT context without touching
        # fake_score (which stays at 3 ctx chunks + 1 live = 4 * npb).
        real_k = int(getattr(args, "real_score_num_gt_chunks", 2))
        real_scorer_num_frames = (real_k + 2) * npb
        fake_scorer_num_frames = 4 * npb
        self.real_score = WanDiffusionWrapper(
            model_name=self.real_model_name, is_causal=False
        )
        # Override the wrapper's spatial-only ``_base_seq_len`` (default
        # 32760 = 21 * 1560) to match the scorer's actual DMD input
        # window. ``adjust_seq_len_for_action_tokens`` reads from
        # ``_base_seq_len`` and adds the per-frame action token budget on
        # top — must happen AFTER this reset.
        self.real_score._base_seq_len = real_scorer_num_frames * 1560
        self.real_score.seq_len = self.real_score._base_seq_len
        if self._action_patch_enabled:
            apply_action_patches_critic(self.real_score)
            self.real_score.model.action_tokens_per_frame = 1
            self.real_score.adjust_seq_len_for_action_tokens(
                num_frames=real_scorer_num_frames, action_per_frame=1,
            )
            if int(getattr(self.real_score.model, "action_tokens_per_frame", 0)) != 1:
                raise RuntimeError(
                    "real_score inner model has action_tokens_per_frame "
                    "!= 1 after Stream B wiring — the DiT will run at "
                    "1560/frame instead of the trained 1561/frame."
                )
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(
            model_name=self.fake_model_name, is_causal=False
        )
        # See the matching comment on ``real_score`` above — size the
        # fake-score wrapper's seq_len to the DMD batched window.
        self.fake_score._base_seq_len = fake_scorer_num_frames * 1560
        self.fake_score.seq_len = self.fake_score._base_seq_len
        if self._action_patch_enabled:
            apply_action_patches_critic(self.fake_score)
            self.fake_score.model.action_tokens_per_frame = 1
            self.fake_score.adjust_seq_len_for_action_tokens(
                num_frames=fake_scorer_num_frames, action_per_frame=1,
            )
            if int(getattr(self.fake_score.model, "action_tokens_per_frame", 0)) != 1:
                raise RuntimeError(
                    "fake_score inner model has action_tokens_per_frame "
                    "!= 1 after Stream B wiring — the DiT will run at "
                    "1560/frame instead of the trained 1561/frame."
                )
        self.fake_score.model.requires_grad_(True)
        self._fake_score_trainable = True

        if not self.text_pre_encoded:
            self.text_encoder = WanTextEncoder(model_name=self.real_model_name)
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder = None

        self.vae = WanVAEWrapper(model_name=self.real_model_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _init_action_projection(self, args, device) -> None:
        if not self._action_patch_enabled:
            self._set_action_projection(None)
            return
        action_dim = int(getattr(args, "raw_action_dim", getattr(args, "action_dim", 2)))
        activation = getattr(args, "action_modulation_activation", None)
        self._action_dim = action_dim
        model_dim = getattr(self.generator.model, "dim", 2048)
        dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        module = ActionModulationProjection(
            action_dim=action_dim,
            activation=activation,
            hidden_dim=model_dim,
            num_frames=1,
            zero_init=True,
        ).to(device=device, dtype=dtype)
        self._set_action_projection(module)
        self._sync_action_projection_params()

    @property
    def action_projection(self) -> Optional[nn.Module]:
        return getattr(self, "_action_projection_ref", None)

    def _set_action_projection(self, module: Optional[nn.Module]) -> None:
        object.__setattr__(self, "_action_projection_ref", module)

    def _sync_action_projection_params(self) -> None:
        module = getattr(self, "_action_projection_ref", None)
        if module is None or not dist.is_initialized():
            return
        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param.data, src=0)

    # ------------------------------------------------------------------
    # Stream B: per-frame action-token projection
    # ------------------------------------------------------------------
    def _init_action_token_projection(self, args, device) -> None:
        """Instantiate the v14/ODE ``ActionTokenProjection`` Stream-B MLP.

        Mirrors ``_init_action_projection`` but emits per-frame action
        tokens (``[B, F, hidden_dim]``) instead of AdaLN modulation. Kept
        outside the DDP-wrapped generator (same as ``action_projection``)
        so its gradients must be manually all-reduced by the trainer.
        """
        if not self._action_patch_enabled:
            self._set_action_token_projection(None)
            return
        action_dim = int(
            getattr(args, "raw_action_dim", getattr(args, "action_dim", 2))
        )
        activation = getattr(args, "action_modulation_activation", None)
        model_dim = getattr(self.generator.model, "dim", 2048)
        dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        module = ActionTokenProjection(
            action_dim=action_dim,
            activation=activation,
            hidden_dim=model_dim,
            zero_init=True,
        ).to(device=device, dtype=dtype)
        self._set_action_token_projection(module)
        self._sync_action_token_projection_params()

    @property
    def action_token_projection(self) -> Optional[nn.Module]:
        return getattr(self, "_action_token_projection_ref", None)

    def _set_action_token_projection(self, module: Optional[nn.Module]) -> None:
        object.__setattr__(self, "_action_token_projection_ref", module)

    def _sync_action_token_projection_params(self) -> None:
        module = getattr(self, "_action_token_projection_ref", None)
        if module is None or not dist.is_initialized():
            return
        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param.data, src=0)

    def _set_fake_score_trainable(self, value: bool) -> None:
        if getattr(self, "_fake_score_trainable", None) == value:
            return
        for param in self.fake_score.parameters():
            param.requires_grad_(value)
        self._fake_score_trainable = value

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep

        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    @staticmethod
    def _to_dict(maybe_cfg) -> dict:
        if maybe_cfg is None:
            return {}
        if isinstance(maybe_cfg, dict):
            return dict(maybe_cfg)
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
        except ImportError:  # pragma: no cover
            DictConfig = None  # type: ignore
            OmegaConf = None  # type: ignore
        else:
            if isinstance(maybe_cfg, DictConfig):
                return OmegaConf.to_container(maybe_cfg, resolve=True)  # type: ignore[return-value]
        if hasattr(maybe_cfg, "items"):
            return {k: v for k, v in maybe_cfg.items()}
        return {}


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        # Lazy-initialized training-time pipeline for backward simulation
        self.inference_pipeline = None

    def _compute_action_modulation_tensor(
        self,
        actions: Optional[torch.Tensor],
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        detach: bool = False,
        frame_start: int = 0,
    ) -> Optional[torch.Tensor]:
        if not self._action_patch_enabled or actions is None:
            raise ValueError("there is an issue with actions")
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        action_inputs = {"actions": actions}
        modulation = self.inference_pipeline._compute_action_modulation(
            action_inputs,
            frame_start=frame_start,
            num_frames=num_frames,
            device=device,
            dtype=dtype,
        )
        if detach and modulation is not None:
            return modulation.detach()
        return modulation

    def _with_action_conditioning(
        self,
        base_conditional: dict,
        actions: Optional[torch.Tensor],
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        detach_modulation: bool = False,
        frame_start: int = 0,
        target_num_frames: Optional[int] = None,
    ) -> dict:
        modulation = self._compute_action_modulation_tensor(
            actions,
            num_frames,
            device,
            dtype,
            detach=detach_modulation,
            frame_start=frame_start,
        )

        conditioned = dict(base_conditional)
        conditioned["_action_modulation"] = modulation
        return conditioned

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 21,
        action_inputs: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int], int]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [min_num_frames, self.num_training_frames], but still being a multiple of self.num_frame_per_block.
        # If `min_num_frames` is not provided, we fallback to the original default behaviour.
        min_num_frames = (self.min_num_training_frames - 1) if self.args.independent_first_frame else self.min_num_training_frames
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if dist.get_rank() == 0 and DEBUG:
            print(f"num_generated_frames: {num_generated_frames}")
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            slice_last_frames=slice_last_frames,
            action_inputs=action_inputs,
            **conditional_dict,
        )
        # Decide whether to slice based on `slice_last_frames`; when `slice_last_frames == -1`, keep all frames
        if slice_last_frames != -1 and pred_image_or_video.shape[1] > slice_last_frames:
            with torch.no_grad():
                # Re-encode: take all frames before the last (slice_last_frames - 1) frames for pixel decoding
                if slice_last_frames > 1:
                    latent_to_decode = pred_image_or_video[:, :-(slice_last_frames - 1), ...]
                else:
                    latent_to_decode = pred_image_or_video
                # Decode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            if slice_last_frames > 1:
                last_frames = pred_image_or_video[:, -(slice_last_frames - 1):, ...]
                pred_image_or_video_sliced = torch.cat([image_latent, last_frames], dim=1)
            else:
                pred_image_or_video_sliced = image_latent
        else:
            pred_image_or_video_sliced = pred_image_or_video

        frame_start_index = max(0, pred_image_or_video.shape[1] - pred_image_or_video_sliced.shape[1])

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_sliced, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_sliced = pred_image_or_video_sliced.to(self.dtype)
        return (
            pred_image_or_video_sliced,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
            frame_start_index,
        )

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        action_inputs: Optional[object] = None,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        kwargs = {
            k: v for k, v in conditional_dict.items() if k != "initial_latent"
        }
        if self._action_patch_enabled:
            return self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                **conditional_dict,
                slice_last_frames=slice_last_frames,
                action_inputs=action_inputs,
            )
        else:
            return self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                **conditional_dict,
                slice_last_frames=slice_last_frames,
            )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        num_training_frames = getattr(self.args, "num_training_frames")
        shared_kwargs = dict(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=num_training_frames,
            context_noise=self.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
        )

        if self._action_patch_enabled:
            self.inference_pipeline = ActionSelfForcingTrainingPipeline(
                **shared_kwargs,
                action_dim=getattr(self.args, "raw_action_dim", getattr(self.args, "action_dim", 2)),
                action_projection=self.action_projection,
            )
        else:
            self.inference_pipeline = SelfForcingTrainingPipeline(**shared_kwargs)
