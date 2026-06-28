#!/usr/bin/env bash
# Submit yearly routing only (no clear / placement). Run from repo root.
# Default: MaxCov + TOPGrowing (6 tasks). Pass argument "linear" for LinearMinTime only (3 tasks).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "linear" ]]; then
  J=$(sbatch --parsable "${SCRIPT_DIR}/supercloud_3_greedy_uniform_routing_linear_array.sh")
  echo "Submitted LinearMinTime routing array: ${J}"
else
  J=$(sbatch --parsable "${SCRIPT_DIR}/supercloud_3_greedy_uniform_routing_array.sh")
  echo "Submitted MaxCov+TOPGrowing routing array: ${J}"
fi
