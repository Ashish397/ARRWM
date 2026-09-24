#!/bin/bash
# Convenience launcher for roll10k step 400 (pointwise MSE).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/ode_panel32_on_holder.sh" pointwise_mse "$@"
