import sys, os, random
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import numpy as np
import torch; torch.set_num_threads(1)
from utils.zarr_dataset import ZarrRideDataset

man = torch.load('logs/v14_balanced_weunz/.ride_manifest.pt', map_location='cpu', weights_only=False)
rides = man['rides'] if isinstance(man, dict) and 'rides' in man else man
print("manifest rides:", len(rides), "type:", type(rides))
random.seed(0)
sample = rides if len(rides) <= 200 else random.sample(rides, 200)

ds = ZarrRideDataset.from_manifest(
    rides_data=sample,
    motion_root='/projects/u6ex/fbots/frodobots_motion',
    ss_vae_checkpoint='action_query/checkpoints/ss_vae_8free.pt',
    device='cpu', ss_vae_device='cpu',
)
n = len(ds)
all_z7 = []
ride_back_frac = []
ok = 0
for i in range(n):
    try:
        r = ds[i]
        zp = r['zarr_path']; nl = int(r['n_latent_frames'])
        if nl < 6:
            continue
        z = ds.encode_z_actions_window(zp, nl, 0, nl)  # [nl, 8]
        z7 = z[:, 7].float().numpy()
        all_z7.append(z7)
        ride_back_frac.append(float((z7 < -0.1).mean()))
        ok += 1
    except Exception as e:
        continue
z7 = np.concatenate(all_z7) if all_z7 else np.array([])
print(f"\nrides analyzed: {ok}/{n} | total latent frames: {len(z7)}")
print(f"z7 (throttle; + forward, - reverse): mean={z7.mean():.3f} median={np.median(z7):.3f} std={z7.std():.3f} min={z7.min():.3f} max={z7.max():.3f}")
for thr in [0.0, -0.05, -0.1, -0.2, -0.3]:
    print(f"  frac z7 < {thr:+.2f} (backward): {100*(z7<thr).mean():.2f}%")
for thr in [0.1, 0.2, 0.3]:
    print(f"  frac z7 > {thr:+.2f} (forward):  {100*(z7>thr).mean():.2f}%")
print(f"  frac |z7| < 0.05 (~stationary):    {100*(np.abs(z7)<0.05).mean():.2f}%")
rbf = np.array(ride_back_frac)
print(f"\nper-ride backward(<-0.1) fraction: mean={100*rbf.mean():.2f}%  median={100*np.median(rbf):.2f}%  max={100*rbf.max():.2f}%")
print(f"rides with >5% backward frames: {int((rbf>0.05).sum())}/{len(rbf)}")
# histogram
hist, edges = np.histogram(z7, bins=np.linspace(-1, 1, 21))
print("\nz7 histogram (bins -1..1):")
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{lo:+.1f},{hi:+.1f}): {'#'*int(60*h/max(hist.max(),1))} {100*h/len(z7):.1f}%")
