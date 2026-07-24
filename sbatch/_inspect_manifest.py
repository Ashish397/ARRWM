import torch
m = torch.load('analysis/eval_final/flow_viz/.rec_scratch/.ride_manifest.pt', map_location='cpu')
print('top keys:', list(m))
for k in m:
    v = m[k]
    print(k, '->', type(v).__name__, (len(v) if hasattr(v,'__len__') else v))
# find the rides container
for k in m:
    v = m[k]
    if isinstance(v, (list, dict)) and k not in ('version','encoded_root'):
        first = (v[0] if isinstance(v, list) else v[list(v)[0]])
        print(f'  {k} elem:', type(first).__name__, list(first)[:8] if isinstance(first, dict) else first)
