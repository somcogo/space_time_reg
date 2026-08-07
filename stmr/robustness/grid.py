"""The robustness sweep matrix: one axis varied at a time from each scene's baseline.

Each GridPoint becomes its own `python -m stmr.ablation run` invocation with a DISTINCT
outdir (build_config forces exp_name=stage.id, so same-scene points would otherwise overwrite
each other's res.pt). The outdir encodes scene/axis/key so collect.py can parse it back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GridPoint:
    scene: str        # "S0" | "S5"  (also the stage id)
    axis: str         # "det" | "seed" | "init" | "noise"
    key: str          # axis-value label, e.g. "r0", "seed3", "ktavg", "0.05"
    overrides: dict = field(default_factory=dict)   # Config --set overrides
    root: str = "."

    @property
    def outdir(self) -> str:
        return os.path.join(self.root, self.scene, self.axis, self.key)


# init-axis recipes (S5 only). nmapg is the S5 baseline (stage already sets it), so it is the
# det/seed origin and not repeated on the init axis.
_INIT_OVERRIDES = {
    "gt": {"init": "gt", "init_skip": True, "use_nmapg": False},
    "adj": {"init": "adj", "init_skip": True, "use_nmapg": False},
    "ktavg": {"init": "ktavg", "init_skip": True, "use_nmapg": False},
}


def build_matrix(root, scenes=("S0", "S5"), seeds=(1, 2, 3, 4),
                 s0_img_sigmas=(0.02, 0.05, 0.10),
                 s5_k_sigmas=(0.02, 0.05, 0.10),
                 s5_inits=("gt", "adj", "ktavg"),
                 det_reps=3) -> list[GridPoint]:
    """Build the standard one-axis-at-a-time sweep. Baselines: S0={seed0,gt-frozen,sigma0},
    S5={seed0,nmapg,sigma0}; the det/r0 run doubles as each axis's origin."""
    pts: list[GridPoint] = []
    for scene in scenes:
        # determinism baseline: identical config, repeated (measures the CUDA-nondeterminism floor)
        for r in range(det_reps):
            pts.append(GridPoint(scene, "det", f"r{r}", {"seed": 0}, root))
        # seed axis
        for s in seeds:
            pts.append(GridPoint(scene, "seed", f"seed{s}", {"seed": s}, root))
        # noise axis: image noise for S0 (k-space term off), k-space noise for S5
        if scene == "S0":
            for sig in s0_img_sigmas:
                pts.append(GridPoint(scene, "noise", f"{sig}",
                                     {"seed": 0, "init_noise_sigma": sig}, root))
        else:
            for sig in s5_k_sigmas:
                pts.append(GridPoint(scene, "noise", f"{sig}",
                                     {"seed": 0, "kspace_noise_sigma": sig}, root))
            # init axis (S5 only)
            for name in s5_inits:
                pts.append(GridPoint(scene, "init", name,
                                     {"seed": 0, **_INIT_OVERRIDES[name]}, root))
    return pts
