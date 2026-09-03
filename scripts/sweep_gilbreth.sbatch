#!/bin/bash
#SBATCH --job-name=echogap-sweep
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gilbreth/gupta596/MotionGen/HOI/echogap/logs/%x-%j.log
# Full EchoGap run: stage A renders (per scene), stage B detector on all sweeps, stage C aggregate.
# Usage: sbatch sweep.sbatch <run_name> <max_scans_per_scene> <scene ...>
set -uxo pipefail
B=/scratch/gilbreth/gupta596/MotionGen/HOI/echogap
source "$B/env.sh"
RUN=$1; NSCANS=$2; shift 2; SCENES="$@"
OUT=$B/outputs/echogap_runs/$RUN
mkdir -p "$OUT/sweeps" "$OUT/results"
OFFSETS="0,0,0 0,1,0 0,2,0 0,3,0 0,0,1"
export APPTAINERENV_PYTHONPATH="$B/pyoverlay:$B/neurad-studio"
for S in $SCENES; do
  if [ "$S" = "001" ]; then CFG=$(ls -t "$B"/outputs/full-default-seq/splatad/*/config.yml | head -1); else CFG=$(ls -t "$B"/outputs/seq$S/splatad/*/config.yml | head -1); fi
  apptainer exec --cleanenv --nv --bind "$B:$B" --pwd "$B/neurad-studio" "$B/neurad-v3.sif" \
    python "$B/echogap-kit/sweep_render.py" "$CFG" "$S" "$OUT/sweeps" "$NSCANS" $OFFSETS
done
export APPTAINERENV_PYTHONPATH="$B/pyoverlay_mm:$B/pyoverlay"
apptainer exec --cleanenv --nv --bind "$B:$B" --pwd "$B" "$B/neurad-v3.sif" \
  python "$B/echogap-kit/detect_bins.py" "$B/checkpoints/centerpoint_nus.pth" "$OUT/detections.json" "$OUT"/sweeps/*.bin
apptainer exec --cleanenv --nv --bind "$B:$B" --pwd "$B" "$B/neurad-v3.sif" \
  python "$B/echogap-kit/aggregate.py" "$OUT/sweeps" "$OUT/detections.json" "$OUT/results"
echo "=== SWEEP EXIT: $? ==="
