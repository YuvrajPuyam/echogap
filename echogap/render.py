"""EchoGap stage A: render every eval scan of one scene at each sensor offset (same beam pattern,
sensor translated in the lidar frame, points expressed back in the ORIGINAL lidar frame) and write
5-dim .bin sweeps + a manifest. Usage: sweep_render.py <config.yml> <scene_id> <out_dir> <max_scans> <dx,dy,dz ...>"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.utils.eval_utils import eval_setup



def main():
    cfg, scene, out, max_scans = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    offsets = [tuple(float(v) for v in o.split(",")) for o in sys.argv[5:]]
    out.mkdir(parents=True, exist_ok=True)
    config, pipeline, ckpt, step = eval_setup(cfg, test_mode="test")
    model = pipeline.model
    model.eval()
    dataset = pipeline.datamanager.eval_lidar_dataset
    loader = FixedIndicesEvalDataloader(dataset=dataset, device=pipeline.device, num_workers=0)
    n_scans = min(max_scans, len(dataset))
    print(f"scene {scene}: step {step}, {len(dataset)} eval scans, rendering {n_scans} x {len(offsets)} offsets")

    manifest = []
    t0 = time.time()
    with torch.no_grad():
        for si in range(n_scans):
            lidar0, batch0 = loader.get_camera(si)
            real = batch0["lidar"][:, :4].cpu().numpy().astype(np.float32)
            if real[:, 3].max() <= 1.0:
                real[:, 3] *= 255.0
            rp = out / f"{scene}_s{si:02d}_real.bin"
            np.concatenate([real, np.zeros((len(real), 1), np.float32)], 1).tofile(rp)
            manifest.append(dict(scene=scene, scan=si, kind="real", offset=[0, 0, 0], bin=str(rp), n_points=int(len(real))))
            for dx, dy, dz in offsets:
                lidar, batch = copy.deepcopy(lidar0), copy.deepcopy(batch0)
                R = lidar.lidar_to_worlds[0, :3, :3]
                shift = torch.tensor([dx, dy, dz], dtype=lidar.lidar_to_worlds.dtype, device=lidar.lidar_to_worlds.device)
                lidar.lidar_to_worlds[0, :3, 3] += R @ shift
                outputs, batch = model.get_outputs_for_lidar(lidar, batch=batch)
                # SplatAD returns a [B, H, W, ...] raster with a validity mask; ray-based models (NeuRAD)
                # return one row per real ray with no mask. Handle both.
                pts_all = outputs["points"].reshape(-1, 3)
                keep = torch.ones(len(pts_all), dtype=torch.bool, device=pts_all.device)
                if "points_valid" in outputs:
                    keep &= outputs["points_valid"].reshape(-1)
                if "ray_drop_prob" in outputs:
                    keep &= outputs["ray_drop_prob"].reshape(-1) < 0.5
                pts = pts_all[keep] + shift
                if "intensity" in outputs:
                    inten = (outputs["intensity"].reshape(-1)[keep] * 255.0).clamp(0, 255)
                else:
                    inten = torch.zeros(len(pts), device=pts.device)
                sweep = torch.cat([pts, inten[:, None], torch.zeros_like(inten)[:, None]], 1).cpu().numpy().astype(np.float32)
                bp = out / f"{scene}_s{si:02d}_dx{dx:g}_dy{dy:g}_dz{dz:g}.bin"
                sweep.tofile(bp)
                manifest.append(dict(scene=scene, scan=si, kind="render", offset=[dx, dy, dz], bin=str(bp), n_points=int(len(sweep)),
                                     nonfinite=int((~np.isfinite(sweep)).sum())))
            print(f"  scan {si}: done ({time.time() - t0:.0f}s elapsed)")
    json.dump(manifest, open(out / f"manifest_{scene}.json", "w"), indent=1)
    print(f"SWEEP_RENDER_OK scene={scene} entries={len(manifest)}")


if __name__ == "__main__":
    main()
