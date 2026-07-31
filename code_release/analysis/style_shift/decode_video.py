"""Decode a video to frames.npy + 1s contact sheet. Usage: decode_video.py <mp4> <out_dir>"""
import os
import sys
import av
import numpy as np
from PIL import Image, ImageDraw

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
c = av.open(src)
s = c.streams.video[0]
fps = float(s.average_rate)
frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
print(f"fps {fps}  size {s.width}x{s.height}  frames {len(frames)}  dur {len(frames)/fps:.1f}s")
np.save(f"{out}/frames.npy", np.stack(frames))

sel = list(range(0, len(frames), max(1, round(fps))))
tiles = []
for i in sel:
    im = Image.fromarray(frames[i]).resize((320, 176))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 64, 16], fill=(0, 0, 0))
    d.text((3, 2), f"t={i/fps:.0f}s", fill=(255, 255, 0))
    tiles.append(im)
cols, rows = 6, (len(tiles) + 5) // 6
sh = Image.new("RGB", (cols * 320, rows * 176))
for k, t in enumerate(tiles):
    sh.paste(t, ((k % cols) * 320, (k // cols) * 176))
sh.save(f"{out}/contact_1s.png")
print(f"FPS={fps}")
