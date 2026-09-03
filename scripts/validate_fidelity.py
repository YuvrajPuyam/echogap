"""End-to-end check of SplatADModel.get_outputs_for_lidar on a trained checkpoint.

Renders the first N eval lidar scans through the fixed sensor API, reports rendered-vs-real
metrics (same formulas as the model's own eval), and saves BEV plots + point clouds.
No plotly/kaleido dependency. Usage: python validate_lidar_fix.py <config.yml> <out_dir> [n_scans]
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.utils.eval_utils import eval_setup

cfg_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
n_scans = int(sys.argv[3]) if len(sys.argv) > 3 else 5
out_dir.mkdir(parents=True, exist_ok=True)

config, pipeline, checkpoint_path, step = eval_setup(cfg_path, test_mode="test")
model = pipeline.model
model.eval()
print(f"checkpoint: {checkpoint_path} (step {step})")
print(f"get_outputs_for_lidar defined on: {model.get_outputs_for_lidar.__qualname__}")

lidar_dataset = pipeline.datamanager.eval_lidar_dataset
loader = FixedIndicesEvalDataloader(dataset=lidar_dataset, device=pipeline.device, num_workers=0)
print(f"eval lidar scans available: {len(lidar_dataset)}; validating {n_scans}")

rows = []
with torch.no_grad():
    for idx, (lidar, batch) in enumerate(loader):
        if idx >= n_scans:
            break
        t0 = time.time()
        outputs, batch = model.get_outputs_for_lidar(lidar, batch=batch)
        torch.cuda.synchronize()
        dt = time.time() - t0

        pred, gt = model.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
        row = {
            "scan": idx,
            "render_s": round(dt, 3),
            "raster_shape": list(outputs["points"].shape),
            "n_gt_returns": int(gt["depth"].numel()),
            "median_range_err_m": float(model.median_l2(pred["median_depth"], gt["depth"])),
            "mean_rel_range_err": float(model.mean_rel_l2(pred["median_depth"], gt["depth"])),
            "intensity_rmse": float(model.rmse(pred["intensity"], gt["intensity"])),
            "ray_drop_acc": float(
                (((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]).sum() / gt["valid"].sum()
            ),
            "n_pred_points": int(pred["median_point_cloud"].shape[0]),
            "n_gt_points": int(gt["point_cloud"].shape[0]),
        }
        rows.append(row)
        print(json.dumps(row))

        p = pred["median_point_cloud"].cpu().numpy()
        g = gt["point_cloud"].cpu().numpy()
        np.savez_compressed(out_dir / f"scan{idx:03d}_points.npz", pred=p, gt=g)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, pts, title in ((axes[0], g, "real scan"), (axes[1], p, "rendered (SplatAD, fixed API)")):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c=np.clip(pts[:, 2], -3, 5), cmap="viridis")
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 80)
            ax.set_aspect("equal")
            ax.set_title(f"{title}: {pts.shape[0]} pts")
        fig.suptitle(f"scan {idx} | median range err {row['median_range_err_m']:.3f} m | step {step}")
        fig.savefig(out_dir / f"scan{idx:03d}_bev.png", dpi=110)
        plt.close(fig)

summary = {
    "checkpoint": str(checkpoint_path),
    "step": step,
    "n_scans": len(rows),
    "median_range_err_m_median": float(np.median([r["median_range_err_m"] for r in rows])),
    "intensity_rmse_mean": float(np.mean([r["intensity_rmse"] for r in rows])),
    "ray_drop_acc_mean": float(np.mean([r["ray_drop_acc"] for r in rows])),
    "rows": rows,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
print("LIDAR_FIX_VALIDATION_OK")
