"""Turn images logged to TensorBoard into per-tag GIFs (training evolution animations).

Reads a run's event files, collects every image logged under the requested tags (one
image per logged epoch; extended logging writes them at epoch 1, every 25th epoch and on
best-loss epochs when ``debug=True``), and writes one GIF per tag with the epoch number
burned into the top-left corner. Duplicate steps -- e.g. from a crashed run's leftover
event file next to the rerun's, or the final-eval re-log of the last epoch -- are deduped
by keeping the latest event per step.

Usage:
  python -m stmr.viz.tb_gif --logdir log/<run>                # flows/vel_col + vel_norm
  python -m stmr.viz.tb_gif --logdir log/<run> --tags flows/flow imgs/moving --fps 4

Outputs ``<tag basename>.gif`` (e.g. vel_col.gif, vel_norm.gif) into ``--out``
(default: <logdir>/gifs).
"""

from __future__ import annotations

import argparse
import io
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = ["flows/vel_col", "flows/vel_norm"]


def resolve_tb_dir(logdir: str) -> str:
    """Accept either a run directory (containing tensorboard/) or the tb dir itself."""
    tb = os.path.join(logdir, "tensorboard")
    return tb if os.path.isdir(tb) else logdir


def load_image_events(tb_dir: str, tag: str):
    """Return the tag's image events as (step, png_bytes), deduped by step (latest wins)
    and sorted by step."""
    acc = EventAccumulator(tb_dir, size_guidance={"images": 0})  # 0 = keep all
    acc.Reload()
    available = acc.Tags()["images"]
    if tag not in available:
        raise SystemExit(f"tag {tag!r} not found; image tags in {tb_dir}: {available}")
    by_step = {}
    for ev in acc.Images(tag):
        if ev.step not in by_step or ev.wall_time >= by_step[ev.step].wall_time:
            by_step[ev.step] = ev
    return [(step, by_step[step].encoded_image_string) for step in sorted(by_step)]


def _label_font(img_h: int):
    size = max(14, img_h // 18)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: fixed-size bitmap font only
        return ImageFont.load_default()


def render_frames(events, label: bool = True) -> list[np.ndarray]:
    """Decode PNG bytes to RGB arrays, burning 'epoch N' into the corner of each frame."""
    frames = []
    for step, png in events:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        if label:
            draw = ImageDraw.Draw(img)
            draw.text((8, 6), f"epoch {step}", fill="white", stroke_width=2,
                      stroke_fill="black", font=_label_font(img.height))
        frames.append(np.asarray(img))
    return frames


def tag_to_gif(tb_dir: str, tag: str, out_dir: str, fps: float, label: bool) -> str:
    events = load_image_events(tb_dir, tag)
    frames = render_frames(events, label)
    out_path = os.path.join(out_dir, tag.rsplit("/", 1)[-1] + ".gif")
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    steps = [s for s, _ in events]
    print(f"{tag}: {len(frames)} frames (epochs {steps[0]}..{steps[-1]}) -> {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logdir", required=True,
                   help="run directory (e.g. log/<run>) or its tensorboard/ subdirectory")
    p.add_argument("--out", default=None,
                   help="output directory for the GIFs (default: <logdir>/gifs)")
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                   help=f"image tags to animate (default: {' '.join(DEFAULT_TAGS)})")
    p.add_argument("--fps", type=float, default=2.0,
                   help="GIF frames per second (default 2; frames are logged epochs)")
    p.add_argument("--no-label", action="store_true",
                   help="do not burn the epoch number into the frames")
    a = p.parse_args(argv)

    tb_dir = resolve_tb_dir(a.logdir)
    out_dir = a.out or os.path.join(a.logdir, "gifs")
    os.makedirs(out_dir, exist_ok=True)
    for tag in a.tags:
        tag_to_gif(tb_dir, tag, out_dir, a.fps, label=not a.no_label)


if __name__ == "__main__":
    main()
