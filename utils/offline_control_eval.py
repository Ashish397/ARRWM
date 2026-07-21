"""Offline GT-vs-FLIP control eval over a run's SAVED checkpoints.

Reuses the trainer's own _control_test_eval (identical rollout + teacher-read +
metric) but drives it from disk: build the trainer once (single process), then
for each checkpoint load its weights via _maybe_resume and run the eval for
ranks 0..N-1 (= N held-out rides/offsets), writing metrics to a SEPARATE logdir
so it never collides with a live run still writing to the original logdir.

Env:
  OCE_CKPT_DIR : dir with causal_lora_step*.pt (default logs/v14e_noatok)
  OCE_CONFIG   : config yaml (default noatok)
  OCE_OUT      : offline logdir (default logs/v14e_noatok_offline)
  OCE_STRIDE   : checkpoint step stride (default 200)
  OCE_MIN/MAX  : step range (default 100 .. 2800)
  OCE_NVID     : rides/offsets per checkpoint (default 16)
"""
import os, re, glob
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
import torch
from omegaconf import OmegaConf
from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer

CKPT_DIR = os.environ.get("OCE_CKPT_DIR", "logs/v14e_noatok")
CONFIG = os.environ.get("OCE_CONFIG", "configs/causal_lora_diffusion_teacher_v14e_noatok.yaml")
OUT = os.environ.get("OCE_OUT", "logs/v14e_noatok_offline")
STRIDE = int(os.environ.get("OCE_STRIDE", "200"))
SMIN = int(os.environ.get("OCE_MIN", "100"))
SMAX = int(os.environ.get("OCE_MAX", "2800"))
NVID = int(os.environ.get("OCE_NVID", "16"))


def main():
    os.makedirs(OUT, exist_ok=True)
    # reuse the live run's manifest cache (symlink) so we don't rebuild for 20min
    for f in (".ride_manifest.pt", ".ride_manifest_shared.pt"):
        src = os.path.abspath(os.path.join(CKPT_DIR, f)); dst = os.path.join(OUT, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
    # CRITICAL: the dataset's action encoder is chosen by this env var (default ss_vae),
    # NOT the config. The live train sbatch exports it; the offline driver must too, or the
    # eval feeds commands in the wrong space (ss_vae) vs the teacher-read (pca_raw) -> scrambled.
    _enc = str(cfg.get("teacher_action_encoder", "ss_vae"))
    os.environ["ARRWM_ACTION_ENCODER"] = _enc
    print(f"[oce] ARRWM_ACTION_ENCODER={_enc} (must match teacher_action_encoder={_enc})", flush=True)
    cfg.logdir = os.path.abspath(OUT)
    cfg.auto_resume = False            # we load each checkpoint manually
    cfg.control_test = True
    cfg.control_eval_start = 0         # no step gating
    cfg.eval_interval = 1
    cfg.control_n_videos = NVID
    cfg.save_checkpoints = False
    cfg.stop_at_step = 0

    print(f"[oce] building trainer (config={CONFIG}, logdir={OUT}) ...", flush=True)
    trainer = CausalLoRADiffusionTrainer(cfg)

    _explicit = os.environ.get("OCE_STEPS", "").strip()
    if _explicit:
        steps = [int(x) for x in _explicit.split(",") if x.strip()]
    else:
        steps = sorted(int(re.search(r"step0*(\d+)", os.path.basename(p)).group(1))
                       for p in glob.glob(f"{CKPT_DIR}/causal_lora_step*.pt"))
        steps = [s for s in steps if SMIN <= s <= SMAX and s % STRIDE == 0]
    print(f"[oce] {len(steps)} checkpoints to eval: {steps}", flush=True)

    for S in steps:
        # EXACT path only -- a glob like *{S}.pt would also match 1100/2100/... for S=100
        ckpt = f"{CKPT_DIR}/causal_lora_step{S:07d}.pt"
        if not os.path.exists(ckpt):
            print(f"[oce] step {S}: checkpoint {ckpt} not found, skip"); continue
        trainer.config.resume_from = ckpt
        trainer.start_step = 0
        trainer._maybe_resume()        # restores LoRA + action_projection + critic
        for r in range(NVID):
            trainer.global_rank = r
            trainer.is_main_process = (r == 0)
            try:
                trainer._control_test_eval(S - 1)   # logs step=S to OUT/control_test/metrics_r{r}.jsonl
            except Exception as e:
                print(f"[oce] step {S} rank {r} failed: {e}")
        print(f"[oce] checkpoint {S} done", flush=True)
    print("[oce] ALL DONE")


if __name__ == "__main__":
    main()
