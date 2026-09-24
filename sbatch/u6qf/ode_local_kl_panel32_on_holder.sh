#!/bin/bash
# Convenience launcher for rollkl10k step 400 (cache-aligned local KL).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/ode_panel32_on_holder.sh" local_kl "$@"
