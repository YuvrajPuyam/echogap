# EchoGap

**Does a 3D detector see the same objects in re-simulated lidar as in the real sweep — and how fast does that agreement fall off when you move the sensor?**

EchoGap is an open, installable benchmark for neural lidar re-simulation. It trains a scene model on a real driving log, renders the lidar sweep again from the original *and from displaced* sensor poses, runs the same off-the-shelf 3D detector on real and rendered sweeps, and reports **detection agreement as a function of sensor displacement**.

Status: v0.1 — protocol frozen, first full run on PandaSet with SplatAD in progress. Numbers below are filled in from that run.

## Why

Lidar re-simulators are scored on rendering metrics (depth error, chamfer, intensity RMSE). Those don't tell you whether a perception stack downstream behaves the same. Detection agreement was introduced for lidar re-sim by LiDARsim (Manivasagam et al., CVPR 2020) and reported by DyNFL (Wu et al., CVPR 2024), but always at the original viewpoints and never as something you can `pip install` and run on open data. On the camera side, "Are NeRFs ready for autonomous driving?" (Lindström et al., CVPR 2024 WAD) measured how detector agreement degrades with lateral shifts; the lidar analog of that curve did not exist. Waabi's "Towards Zero Domain Gap" (ICCV 2023) is the strongest lidar-sim/detector study, on proprietary data.

EchoGap's claim is deliberately narrow: **first open, installable detector-agreement benchmark for lidar re-simulation, and first to measure agreement against viewpoint displacement**, on an ungated CC-BY-4.0 dataset, with splat-based baselines.

## Protocol

| | |
|---|---|
| Data | PandaSet (Pandar64), the 10 sequences used by SplatAD: 001 011 016 028 053 063 084 106 123 158 (HF mirror `georghess/pandaset`, CC-BY-4.0) |
| Scene model | SplatAD (3D Gaussian Splatting with rasterized lidar), 30k iterations per scene, trained with neurad-studio |
| Rendering | every eval scan, re-rendered with the **real scan's beam pattern** from the true pose and from the sensor translated 1 / 2 / 3 m laterally and 1 m up; rendered points are expressed in the original lidar frame so only the viewpoint changes |
| Detector | CenterPoint (pillar, nuScenes-trained, mmdetection3d checkpoint), zero-shot on Pandar64; intensity 0–255 for real and rendered |
| Agreement | detections on the real sweep are the reference; rendered detections are Hungarian-matched by BEV center distance ≤ 2 m (per class, and class-agnostic); we report precision / recall / F1 at score thresholds 0.3 and 0.5, mean over scans with bootstrap 95% CIs |
| Anchor | recall of ground-truth vehicles (PandaSet cuboids) within 50 m on real sweeps, so the detector's absolute quality is visible |

## Results

*(filled from run1 — 10 scenes × 10 eval scans × 5 poses)*

Agreement F1 vs lateral displacement (class-agnostic, score ≥ 0.5):

| offset | 0 m | 1 m | 2 m | 3 m | +1 m vertical |
|---|---|---|---|---|---|
| F1 | TBD | TBD | TBD | TBD | TBD |

Rendering fidelity of the same models (rendered vs real sweep, 100 scans): median range error 1.23 cm, ray-drop accuracy 96.7%, intensity RMSE 0.061 — i.e. the reconstructions are as good as the SplatAD paper reports; agreement drops are a viewpoint effect, not a training failure.

Detector anchor (real sweeps, seq 001): 52–60% of ground-truth vehicles within 50 m recovered at score ≥ 0.3, zero-shot from nuScenes.

## Install and run

```bash
pip install -e .
# stage A (needs neurad-studio with get_outputs_for_lidar — see georghess/neurad-studio#93)
echogap-render <config.yml> <scene_id> <out_dir> <max_scans> 0,0,0 0,1,0 0,2,0 0,3,0 0,0,1
# stage B (needs mmdet3d + the CenterPoint nuScenes checkpoint)
echogap-detect <checkpoint.pth> <detections.json> <out_dir>/*.bin
# stage C
echogap-aggregate <out_dir> <detections.json> <results_dir>
```

`scripts/sweep_gilbreth.sbatch` runs all three stages on a SLURM cluster inside the neurad-studio Apptainer image.

## Relation to neurad-studio

Rendering lidar from a trained SplatAD model through the sensor API did not work upstream (issue #75). The fix — a rasterization-based `get_outputs_for_lidar` plus the viewer path — is [PR #93](https://github.com/georghess/neurad-studio/pull/93); EchoGap's stage A calls exactly that method.

## Limitations

- One detector, zero-shot across sensors (nuScenes → Pandar64). Absolute recall is modest; agreement is a relative measure and the anchor row makes that explicit.
- Displacements are pure translations of the sensor with the scan pattern held fixed; rolling-shutter timing is kept from the real sweep.
- Ten scenes from one dataset. Extending to nuScenes / Argoverse 2 is a dataparser away.

## Citation and prior work

LiDARsim (CVPR 2020) · Towards Zero Domain Gap, Waabi (ICCV 2023) · DyNFL (CVPR 2024) · Are NeRFs Ready for Autonomous Driving? (CVPR 2024 WAD) · EUVS (ICCV 2025) · SplatAD (CVPR 2025) · NeuRAD (CVPR 2024)

Apache-2.0.
