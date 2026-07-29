#!/bin/bash
cd /home/ashish/ARRWM/grids/eval
source ~/refmetrics_venv/bin/activate 2>/dev/null
# wait for the DINO style run to release the GPU (avoid concurrent GPU jobs)
while pgrep -f fleet_style_6s.py >/dev/null 2>&1; do sleep 20; done
echo "[chain] DINO done -> starting CoTracker $(date +%H:%M:%S)"
python stationary_cotracker.py && echo "[chain] cotracker done -> wedges" && python stationary_wedges.py
echo "[chain] DONE $(date +%H:%M:%S)"
