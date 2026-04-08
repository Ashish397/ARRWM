# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import types
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Optional, Union
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel

# Load default config to get wan_model_path
default_config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
_default_config = OmegaConf.load(default_config_path)
_default_wan_model_path = _default_config.get("wan_model_path", "/scratch/u6ej/as1748.u6ej/frodobots")
# Ensure path ends with a slash
if not _default_wan_model_path.endswith("/"):
    _default_wan_model_path = _default_wan_model_path + "/"

class ResidualMLPBlock(nn.Module):
    """Residual block for classification head with dropout."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual  # Residual connection

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

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
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

    def adding_cls_branch(
        self, 
        atten_dim=1536, 
        num_class=1, 
        hidden_dim=3072,
        num_layers=4,
        dropout=0.2,
        gan_blocks_per_token=2,
    ) -> None:
        """
        Add classification branch with deeper layers, dropout, and residual connections.
        
        Args:
            atten_dim: Attention dimension (default 1536)
            num_class: Number of output classes (default 1 for binary real/fake)
            hidden_dim: Hidden dimension for intermediate layers (default 3072)
            num_layers: Number of residual layers in classification head (default 4)
            dropout: Dropout rate (default 0.2)
            gan_blocks_per_token: Number of GanAttentionBlocks to stack per register token (default 2).
                                 Using multiple blocks allows progressive refinement of features
                                 from each transformer layer, improving feature extraction.
        """
        # Multi-scale feature extraction: default to extracting from more layers
        layer_indices = [7, 13, 21, 29]
        num_registers = len(layer_indices)
                
        # Input dimension: num_registers * atten_dim
        input_dim = num_registers * atten_dim
        
        # Build deeper classification head with residual connections
        layers = []
        
        # Initial projection and normalization
        layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        
        # Residual blocks
        for _ in range(num_layers - 1):  # -1 because we have final layer
            layers.append(ResidualMLPBlock(hidden_dim, dropout=dropout))
        
        # Final output layer (no residual connection)
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, num_class))
        
        self._cls_pred_branch = nn.Sequential(*layers)
        self._cls_pred_branch.requires_grad_(True)
        
        # Register tokens for each layer we extract from
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        # Stack multiple GAN cross-attention blocks per token for richer feature extraction
        # Structure: ModuleList[ModuleList[GanAttentionBlock]] - one ModuleList per token
        gan_ca_blocks = []
        for _ in range(num_registers):
            token_blocks = []
            for _ in range(gan_blocks_per_token):
                block = GanAttentionBlock()
                token_blocks.append(block)
            gan_ca_blocks.append(nn.ModuleList(token_blocks))
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def adding_rgs_branch(
        self,
        atten_dim: int = 1536,
        num_class: int = 2,
        time_embed_dim: int = 0,
        num_frames: int = 21,
    ) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._rgs_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(1536, num_class * num_frames)
        )
        self._rgs_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens_rgs = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens_rgs.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks_rgs = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks_rgs.requires_grad_(True)
        # self.has_rgs_branch = True

        self.num_frames = num_frames
        self.num_class = num_class

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
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
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
        classify_mode: Optional[bool] = False,
        regress_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None
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

        logits = None
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
            if isinstance(model_out, tuple):
                flow_pred = model_out[0].permute(0, 2, 1, 3, 4)
                state_hidden = model_out[1]
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
        elif classify_mode:
            flow_pred, logits = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                classify_mode=True,
                register_tokens=self._register_tokens,
                cls_pred_branch=self._cls_pred_branch,
                gan_ca_blocks=self._gan_ca_blocks,
                concat_time_embeddings=concat_time_embeddings,
                **action_mod_kwargs,
            )
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        elif regress_mode:
            flow_pred, logits = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                regress_mode=True,
                register_tokens_rgs=self._register_tokens_rgs,
                rgs_pred_branch=self._rgs_pred_branch,
                gan_ca_blocks_rgs=self._gan_ca_blocks_rgs,
                num_frames_rgs=self.num_frames,
                num_class_rgs=self.num_class,
                concat_time_embeddings=concat_time_embeddings,
                **action_mod_kwargs,
            )
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
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
            state_preds, probe_hidden = self._state_probe(
                tapped_features, noisy_start, frame_seqlen,
            )
            return flow_pred, pred_x0, state_preds.float(), probe_hidden

        # State-token readout: pool per chunk and map to z2/z7
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

        if logits is not None:
            return flow_pred, pred_x0, logits

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
