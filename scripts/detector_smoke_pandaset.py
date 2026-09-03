"""Detector-tier smoke test: nuScenes-trained CenterPoint on real PandaSet Pandar64 sweeps.
Decides the EchoGap metric design. Usage: python centerpoint_smoke.py <seq_dir> <ckpt> <out_dir> [frames...]"""
import gzip, json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq, ckpt, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
frames = sys.argv[4:] or ["00", "20", "40"]
out.mkdir(parents=True, exist_ok=True)
poses = json.load(open(seq / "lidar/poses.json"))

def quat_to_R(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_lidar(p_w, pose):
    R = quat_to_R(pose["heading"]); t = np.array([pose["position"][k] for k in "xyz"])
    return (p_w - t) @ R  # R^T applied on the right

def load_sweep(frame, sub32=False):
    df = pickle.load(gzip.open(seq / f"lidar/{frame}.pkl.gz"))
    df = df[df["d"] == 0]  # Pandar64 only
    p = world_to_lidar(df[["x", "y", "z"]].to_numpy(), poses[int(frame)])
    inten = df["i"].to_numpy()
    if sub32:  # approximate 32-beam: bin elevation into 64 channels, keep even ones
        r = np.linalg.norm(p, axis=1); elev = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-6), -1, 1)))
        ch = np.floor((elev - elev.min()) / (elev.max() - elev.min() + 1e-6) * 64).astype(int)
        keep = ch % 2 == 0; p, inten = p[keep], inten[keep]
    return np.concatenate([p, inten[:, None], np.zeros((len(p), 1))], 1).astype(np.float32)

def load_gt(frame):
    c = pickle.load(gzip.open(seq / f"annotations/cuboids/{frame}.pkl.gz"))
    if frame == frames[0]: print("cuboid columns:", list(c.columns)[:14])
    pos = world_to_lidar(c[["position.x", "position.y", "position.z"]].to_numpy(), poses[int(frame)])
    return pos, c["label"].to_numpy(), c[["dimensions.x", "dimensions.y"]].to_numpy()

from mmdet3d.apis import LidarDet3DInferencer
inf = LidarDet3DInferencer(model="centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d", weights=ckpt, device="cuda:0")
CAR_LIKE = {"Car", "Pickup Truck", "Medium-sized Truck", "Semi-truck", "Bus", "Other Vehicle - Construction Vehicle", "Other Vehicle - Uncommon", "Other Vehicle - Pedicab", "Emergency Vehicle"}
summary = []
for frame in frames:
    gt_pos, gt_lab, gt_dim = load_gt(frame)
    for variant in ("full64", "sub32"):
        pts = load_sweep(frame, sub32=(variant == "sub32"))
        binp = out / f"{frame}_{variant}.bin"; pts.tofile(binp)
        res = inf(dict(points=str(binp)), return_datasamples=True, show=False)
        pi = res["predictions"][0].pred_instances_3d
        boxes = pi.bboxes_3d.tensor.cpu().numpy(); scores = pi.scores_3d.cpu().numpy(); labels = pi.labels_3d.cpu().numpy()
        for thr in (0.3, 0.5):
            m = scores >= thr
            # GT cars within 50 m, matched by 2 m center distance (nuScenes convention)
            gm = np.array([l in CAR_LIKE for l in gt_lab]) & (np.linalg.norm(gt_pos[:, :2], axis=1) < 50)
            det_c = boxes[m][:, :2]; car_pred = det_c[labels[m] <= 4]  # nus classes 0-4 = car,truck,trailer,bus,construction
            hit = 0
            for g in gt_pos[gm][:, :2]:
                if len(car_pred) and np.min(np.linalg.norm(car_pred - g, axis=1)) < 2.0: hit += 1
            row = dict(frame=frame, variant=variant, thr=thr, n_points=int(len(pts)), n_det=int(m.sum()),
                       n_gt_cars_50m=int(gm.sum()), gt_cars_recalled=hit, recall=round(hit / max(1, gm.sum()), 3),
                       n_car_pred=int(len(car_pred)))
            summary.append(row); print(json.dumps(row))
        # BEV plot at thr 0.3
        m = scores >= 0.3
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c="lightgray")
        for b in boxes[m]:
            x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
            c, s_ = np.cos(yaw), np.sin(yaw)
            corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
            corners = corners @ np.array([[c, s_], [-s_, c]]) + [x, y]
            ax.plot(corners[:, 0], corners[:, 1], "r-", lw=0.8)
        for g, lab in zip(gt_pos, gt_lab):
            if lab in CAR_LIKE: ax.plot(g[0], g[1], "b+", ms=6)
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
        ax.set_title(f"seq {seq.name} frame {frame} {variant}: CenterPoint(nuScenes) dets (red) vs GT car centers (blue+)")
        fig.savefig(out / f"{frame}_{variant}_bev.png", dpi=100); plt.close(fig)
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print("CENTERPOINT_SMOKE_OK")
