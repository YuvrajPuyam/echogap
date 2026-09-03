"""Run CenterPoint on 5-dim .bin sweeps; save boxes/scores/labels per file.
Usage: detect_bins.py <ckpt> <out_json> <bin...>"""
import json
import sys

import numpy as np
from mmdet3d.apis import LidarDet3DInferencer



def main():
    ckpt, out_json, bins = sys.argv[1], sys.argv[2], sys.argv[3:]
    inf = LidarDet3DInferencer(
        model="centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d", weights=ckpt, device="cuda:0"
    )
    res_all = {}
    for b in bins:
        r = inf(dict(points=b), return_datasamples=True, show=False)["predictions"][0].pred_instances_3d
        res_all[b] = dict(
            boxes=r.bboxes_3d.tensor.cpu().numpy().tolist(),
            scores=r.scores_3d.cpu().numpy().tolist(),
            labels=r.labels_3d.cpu().numpy().tolist(),
        )
        s = np.array(res_all[b]["scores"])
        print(f"{b.split('/')[-1]}: {int((s >= 0.3).sum())} dets @0.3, {int((s >= 0.5).sum())} @0.5")
    json.dump(res_all, open(out_json, "w"))
    print("DETECT_BINS_OK")


if __name__ == "__main__":
    main()
