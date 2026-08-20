import torch, time, resource, sys
t0 = time.time()
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
dt = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"load {dt:.1f}s | rides {len(m['rides'])} | peak RSS {rss:.2f} GB", flush=True)
