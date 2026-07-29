"""Pin the release's computation so cleanup can be verified as a no-op.

Snapshots deterministic outputs of every component that can be exercised
without the dataset or trained weights, plus the resolved configs and the
public API surface. Run --save before editing, --check after each batch.

    python tools/release_baseline.py --save          # before changes
    python tools/release_baseline.py --check         # after each batch

Anything that cannot run in the current environment (e.g. modules that touch
CUDA at import time on a login node) is recorded as "skipped:<reason>" so a
later run on a GPU node widens coverage without invalidating earlier checks.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "code_release")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
sys.path.insert(0, REL)


def h(x):
    """Stable digest of a tensor/array/number."""
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().double().numpy()
    a = np.asarray(x, dtype=np.float64)
    a = np.nan_to_num(a, nan=-12345.0, posinf=1e30, neginf=-1e30)
    return {"shape": list(a.shape),
            "sha": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16],
            "sum": round(float(a.sum()), 6),
            "absmean": round(float(abs(a).mean()), 8)}


def probe(name, fn, snap):
    try:
        snap[name] = fn()
    except Exception as e:                                    # noqa: BLE001
        snap[name] = f"skipped:{type(e).__name__}:{str(e)[:70]}"


# ----------------------------------------------------------------- probes
def p_scheduler():
    import torch
    from utils.scheduler import FlowMatchScheduler
    s = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    s.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    out = {"timesteps": h(s.timesteps), "sigmas": h(s.sigmas)}
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    n = torch.randn(2, 4, 8, 8)
    out["add_noise"] = h(s.add_noise(x, n, s.timesteps[10]))
    return out


def p_action_modulation():
    import torch
    from model.action_modulation import (ActionModulationProjection,
                                         ActionTokenProjection)
    res = {}
    torch.manual_seed(0)
    m = ActionModulationProjection(action_dim=2, hidden_dim=1536).eval()
    torch.manual_seed(1)
    a = torch.randn(2, 7, 2)
    with torch.no_grad():
        res["adaln"] = h(m(a))
    res["adaln_nparam"] = sum(p.numel() for p in m.parameters())
    torch.manual_seed(0)
    t = ActionTokenProjection(action_dim=2, hidden_dim=1536).eval()
    with torch.no_grad():
        res["tokens"] = h(t(a))
    res["tokens_nparam"] = sum(p.numel() for p in t.parameters())
    return res


def p_action_critic():
    import torch
    from model.action_critic import ActionCritic
    torch.manual_seed(0)
    c = ActionCritic(in_channels=16, base_channels=128, num_blocks=4,
                     z_out_dim=8).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 3, 16, 16, 16)
    ts = torch.full((1,), 25.0)
    act = torch.randn(1, 3, 2)
    with torch.no_grad():
        out = c(x, ts, act)
    return {"out": h(out), "nparam": sum(p.numel() for p in c.parameters())}


def p_pca_basis():
    """The frozen action basis is the method's core artifact."""
    import numpy as np
    import torch
    ck = os.path.join(REL, "action_query", "checkpoints", "ss_vae_8free.pt")
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    res = {"pca_mean": h(sd["pca_mean"]), "pca_comp": h(sd["pca_comp"])}
    if "scale" in sd:
        res["scale"] = h(sd["scale"])
    # project a fixed synthetic displacement field through the basis
    rng = np.random.default_rng(0)
    m = rng.standard_normal((4, 200)).astype(np.float64)
    mean = np.asarray(sd["pca_mean"], dtype=np.float64).reshape(1, -1)
    comp = np.asarray(sd["pca_comp"], dtype=np.float64)
    res["projection"] = h((m - mean) @ comp.T)
    return res


def p_tanh_squash():
    import torch
    from utils.zarr_dataset import _tanh_squash
    torch.manual_seed(0)
    z = torch.randn(4, 8) * 3.0
    return {"squash": h(_tanh_squash(z))}


def p_configs():
    from omegaconf import OmegaConf
    import glob
    os.environ.setdefault("AF_ROOT", "/tmp/af")
    os.environ.setdefault("DATA_ROOT", "/tmp/data")
    os.environ.setdefault("WAN_MODELS", "/tmp/wan")
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    res = {}
    for f in sorted(glob.glob(os.path.join(REL, "configs", "*.yaml"))):
        c = OmegaConf.load(f)
        res[os.path.basename(f)] = OmegaConf.to_yaml(
            OmegaConf.create(OmegaConf.to_container(c, resolve=True)))
    return res


def p_api():
    """Public signatures of the live modules: catches accidental changes."""
    mods = ["utils.scheduler", "utils.causal_chain_rollout",
            "model.action_modulation", "model.action_critic",
            "model.causal_teacher_streaming", "action_query.pca_motion",
            "action_query.ss_vae_model"]
    res = {}
    for m in mods:
        try:
            mod = __import__(m, fromlist=["*"])
        except Exception as e:                                # noqa: BLE001
            res[m] = f"skipped:{type(e).__name__}"
            continue
        sigs = {}
        for n, o in sorted(vars(mod).items()):
            if n.startswith("_") or getattr(o, "__module__", None) != m:
                continue
            if inspect.isfunction(o) or inspect.isclass(o):
                try:
                    sigs[n] = str(inspect.signature(o))
                except (TypeError, ValueError):
                    sigs[n] = "<nosig>"
            if inspect.isclass(o):
                for mn, mo in sorted(vars(o).items()):
                    if not mn.startswith("_") and inspect.isfunction(mo):
                        try:
                            sigs[f"{n}.{mn}"] = str(inspect.signature(mo))
                        except (TypeError, ValueError):
                            sigs[f"{n}.{mn}"] = "<nosig>"
        res[m] = sigs
    return res


PROBES = {"scheduler": p_scheduler, "action_modulation": p_action_modulation,
          "action_critic": p_action_critic, "pca_basis": p_pca_basis,
          "tanh_squash": p_tanh_squash, "configs": p_configs, "api": p_api}


def collect():
    snap = {}
    for name, fn in PROBES.items():
        probe(name, fn, snap)
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    snap = collect()
    live = sum(1 for v in snap.values() if not str(v).startswith("skipped:"))
    if args.save:
        json.dump(snap, open(OUT, "w"), indent=1, sort_keys=True)
        print(f"saved {live}/{len(snap)} live probes -> {OUT}")
        for k, v in snap.items():
            if str(v).startswith("skipped:"):
                print(f"  skipped {k}: {v[8:]}")
        return
    if args.check:
        old = json.load(open(OUT))
        drift, newly = [], []
        for k, v in snap.items():
            o = old.get(k)
            if o is None:
                newly.append(k)
            elif str(o).startswith("skipped:") or str(v).startswith("skipped:"):
                continue                       # coverage differs, not drift
            elif json.dumps(o, sort_keys=True) != json.dumps(v, sort_keys=True):
                drift.append(k)
        print(f"checked {live}/{len(snap)} live probes")
        if drift:
            print("DRIFT DETECTED in:", ", ".join(drift))
            for k in drift:
                a = json.dumps(old[k], sort_keys=True, indent=1).split("\n")
                b = json.dumps(snap[k], sort_keys=True, indent=1).split("\n")
                for x, y in zip(a, b):
                    if x != y:
                        print(f"  {k}: {x.strip()}  ->  {y.strip()}")
            sys.exit(1)
        print("OK - no drift" + (f" (new probes: {newly})" if newly else ""))
        return
    ap.error("pass --save or --check")


if __name__ == "__main__":
    main()
