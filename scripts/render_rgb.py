"""Camera-view fly-through of a trained scene: the driven path for one camera, optionally with the
camera moved in its OWN frame (x = right, y = up), which is what "2 m lateral" means elsewhere in
EchoGap. nerfstudio's `render interpolate --shift` adds the offset in world axes, so it is not used.
Usage: render_rgb.py <config.yml> <out.mp4> <sensor_idx> <dx,dy,dz> [outputs=rgb] [fps=10]"""
import sys
from pathlib import Path

import torch

from nerfstudio.scripts.render import _render_trajectory_video, streamline_ad_config
from nerfstudio.utils.eval_utils import eval_setup


def main():
    cfg, out, sensor = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    shift = [float(v) for v in sys.argv[4].split(",")]
    names = sys.argv[5].split(",") if len(sys.argv) > 5 else ["rgb"]
    fps = int(sys.argv[6]) if len(sys.argv) > 6 else 10  # PandaSet cameras run at 10 Hz
    out.parent.mkdir(parents=True, exist_ok=True)

    _, pipeline, _, _ = eval_setup(cfg, test_mode="test", update_config_callback=streamline_ad_config)
    dm = pipeline.datamanager
    cams = dm.train_dataset.cameras.cat([dm.eval_dataset.cameras])
    cams = cams[(cams.metadata["sensor_idxs"] == sensor).squeeze(-1)]
    cams = cams[torch.argsort(cams.times.squeeze(-1))]
    print(f"{cfg.parent.name}: sensor {sensor}, {len(cams)} frames, shift {shift} (camera frame)")

    c2w = cams.camera_to_worlds.clone()
    c2w[..., :3, 3] += c2w[..., :3, :3] @ torch.tensor(shift, dtype=c2w.dtype)
    cams.camera_to_worlds = c2w

    # Render the recorded frames, not an interpolated path: SplatAD's camera model needs per-frame
    # metadata (local linear/angular velocities for rolling shutter) that interpolated Cameras lack.
    path = cams

    _render_trajectory_video(
        pipeline,
        path,
        output_filename=out,
        rendered_output_names=names,
        rendered_resolution_scaling_factor=1.0,
        seconds=len(path) / fps,
        output_format="video",
    )


if __name__ == "__main__":
    main()
