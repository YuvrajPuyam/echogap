#!/bin/bash
#SBATCH --job-name=eg-rgb
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gilbreth/gupta596/hoi/eg/logs/%x-%j.log
# Camera-view fly-throughs of the trained SplatAD scenes, three clips per scene:
#   path   = driven path, front camera, RGB
#   lat2m  = same path, camera 2 m to the right (the benchmark's lateral shift)
#   depth  = driven path, RGB above depth
# Reads checkpoints from the existing echogap folder, writes only under hoi/eg.
# Usage: sbatch render_rgb_gilbreth.sbatch [scene ...]   (default: 001 016 028)
set -uxo pipefail
B=/scratch/gilbreth/gupta596/MotionGen/HOI/echogap
O=/scratch/gilbreth/gupta596/hoi/eg/rgb
mkdir -p "$O"
source "$B/env.sh"
export APPTAINERENV_PYTHONPATH="$B/pyoverlay:$B/neurad-studio"
SCENES="${@:-001 016 028}"
CAM=0   # PandaSet front_camera; check the "sensor 0" line in the log names the front view
RUN() { apptainer exec --cleanenv --nv --bind "$B:$B" --bind /scratch/gilbreth/gupta596/hoi/eg:/scratch/gilbreth/gupta596/hoi/eg --pwd "$B/neurad-studio" "$B/neurad-v3.sif" \
  python /scratch/gilbreth/gupta596/hoi/eg/render_rgb.py "$@"; }
for S in $SCENES; do
  if [ "$S" = "001" ]; then CFG=$(ls -t "$B"/outputs/full-default-seq/splatad/*/config.yml | head -1); else CFG=$(ls -t "$B"/outputs/seq$S/splatad/*/config.yml | head -1); fi
  RUN "$CFG" "$O/seq${S}_path.mp4"  $CAM 0,0,0
  RUN "$CFG" "$O/seq${S}_lat2m.mp4" $CAM 2,0,0
  RUN "$CFG" "$O/seq${S}_depth.mp4" $CAM 0,0,0 rgb,depth
done
ls -la "$O"
echo "=== RGB EXIT ==="
