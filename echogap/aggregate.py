"""EchoGap stage C: detection agreement per (scene, scan, offset) from manifests + detections, then
aggregate per offset with bootstrap CIs and draw the agreement-vs-displacement curve.
Usage: aggregate.py <run_dir> <detections.json> <out_dir>"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

run_dir, det_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
dets = json.load(open(det_path))
manifest = []
for m in glob.glob(str(run_dir / "manifest_*.json")):
    manifest += json.load(open(m))
DIST = 2.0
THRS = (0.3, 0.5)
MODES = ("class", "any")


def boxes(entry, thr, mode):
    b, s, l = np.array(entry["boxes"]), np.array(entry["scores"]), np.array(entry["labels"])
    m = s >= thr
    if mode == "any":
        l = np.zeros_like(l)
    return b[m], l[m]


def agree(ref, oth):
    (rb, rl), (ob, ol) = ref, oth
    tp = 0
    for cls in np.unique(np.concatenate([rl, ol])):
        r, o = rb[rl == cls], ob[ol == cls]
        if len(r) and len(o):
            d = np.linalg.norm(r[:, None, :2] - o[None, :, :2], axis=-1)
            ri, oi = linear_sum_assignment(d)
            tp += int((d[ri, oi] <= DIST).sum())
    rec = tp / len(rb) if len(rb) else np.nan
    prec = tp / len(ob) if len(ob) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(n_real=len(rb), n_render=len(ob), matched=tp, recall=rec, precision=prec, f1=f1)


real_by = {(m["scene"], m["scan"]): m["bin"] for m in manifest if m["kind"] == "real"}
rows = []
for m in manifest:
    if m["kind"] != "render":
        continue
    rb = real_by[(m["scene"], m["scan"])]
    if rb not in dets or m["bin"] not in dets:
        continue
    for thr in THRS:
        for mode in MODES:
            r = agree(boxes(dets[rb], thr, mode), boxes(dets[m["bin"]], thr, mode))
            rows.append(dict(scene=m["scene"], scan=m["scan"], offset=tuple(m["offset"]), thr=thr, mode=mode, **r))
json.dump(rows, open(out / "agreement_rows.json", "w"), indent=1)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


summary = {}
lines = ["| offset (m) | thr | match | scans | mean F1 [95% CI] | recall | precision |", "|---|---|---|---|---|---|---|"]
for thr in THRS:
    for mode in MODES:
        for off in sorted({r["offset"] for r in rows}):
            sel = [r for r in rows if r["thr"] == thr and r["mode"] == mode and r["offset"] == off]
            f1 = boot_ci([r["f1"] for r in sel])
            rec = np.nanmean([r["recall"] for r in sel])
            prec = np.nanmean([r["precision"] for r in sel])
            summary[f"thr{thr}_{mode}_{off}"] = dict(offset=off, thr=thr, mode=mode, n=len(sel), f1=f1[0], f1_lo=f1[1], f1_hi=f1[2], recall=rec, precision=prec)
            lines.append(f"| {off} | {thr} | {mode} | {len(sel)} | {f1[0]:.3f} [{f1[1]:.3f}, {f1[2]:.3f}] | {rec:.3f} | {prec:.3f} |")
md = "\n".join(lines)
open(out / "summary.md", "w").write("# EchoGap: real-vs-rendered detection agreement vs sensor displacement\n\n" + md + "\n")
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(md)

# hero curve: F1 vs lateral offset (dz=0), per thr/mode
fig, ax = plt.subplots(figsize=(7, 5))
for thr in THRS:
    for mode in MODES:
        pts = [(o[1], summary[f"thr{thr}_{mode}_{o}"]) for o in sorted({r["offset"] for r in rows}) if o[0] == 0 and o[2] == 0]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1]["f1"] for p in pts]
        lo = [p[1]["f1_lo"] for p in pts]
        hi = [p[1]["f1_hi"] for p in pts]
        ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], marker="o", capsize=3, label=f"thr {thr}, {mode}")
ax.set_xlabel("lateral sensor displacement (m)")
ax.set_ylabel("detection agreement F1 (real vs rendered)")
ax.set_title("SplatAD lidar re-simulation: detector agreement vs displacement (PandaSet)")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out / "agreement_vs_displacement.png", dpi=120)
print("AGGREGATE_OK")
