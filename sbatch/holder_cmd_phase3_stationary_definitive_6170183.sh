#!/bin/bash
set -euo pipefail
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=6170183 NODE=nid010221 \
RUNSTAMP=2808phase3stationarydefinitive_h6170183 \
bash sbatch/run_phase3_stationary_definitive_node.sh
