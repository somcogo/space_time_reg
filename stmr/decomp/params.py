"""Parameters for the two-stage motion-decomposition experiment.

A flat dataclass holding every knob, kept separate from the 61-field pipeline ``Config``
(which only surfaces transiently in data.py to fetch the GT cine). Mirrors the local-spec
convention of stmr/robustness/grid.py and stmr/ablation/stages.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class DecompParams:
    # --- data ---
    patient: str = "P001"          # dataset = f"cmr_{patient}"
    slice_number: int = 0
    start_frame: int = 0
    time_points: int = 12          # groups = time_points - 1 interval flows

    # --- architectures (last_init_zero=True, omega below, fixed in build_net) ---
    phi1_layers: list = field(default_factory=lambda: [2, 64, 64, 2])        # small / regular
    phi2_layers: list = field(default_factory=lambda: [2, 256, 256, 256, 2])  # large / residual
    omega: float = 30.0

    # --- ODE integration ---
    solver: str = "euler"
    step_size: float = 0.1         # 0 -> adaptive (None)

    # --- optimisation (per run) ---
    steps1: int = 1000
    steps2: int = 1000
    steps_base_large: int = 1000
    steps_base_small: int = 1000
    lr1: float = 3e-4
    lr2: float = 3e-4
    lr_base_large: float = 3e-4
    lr_base_small: float = 3e-4

    # --- regularisation lambdas (data MSE is O(1e-4) after unit-peak normalisation, so these
    #     are moderate nudges; phi1's regularity comes mainly from its small size) ---
    lam_grad_phi1: float = 1e-2    # regular coarse flow (regular but still free to move)
    lam_logdetJ1: float = 1e-2
    lam_grad_phi2: float = 0.0     # free residual on phi2
    lam_logdetJ2: float = 0.0
    lam_grad_base_large: float = 0.0
    lam_logdetJ_base_large: float = 0.0
    lam_grad_base_small: float = 1e-2   # small-alone mirrors phi1's regularisation
    lam_logdetJ_base_small: float = 1e-2

    # --- bookkeeping ---
    exp_name: str = "decomp_p001_s0"
    log_path: str = "log/decomp"
    device: str = "cuda:0"
    seed: int = 0
    log_every: int = 50            # TB scalar / eval cadence (steps)
    run_baselines: bool = False    # only the phi1->phi2 case by default

    @property
    def dataset(self) -> str:
        return f"cmr_{self.patient}"

    @property
    def groups(self) -> int:
        return self.time_points - 1

    def smoke(self) -> "DecompParams":
        """A tiny, CPU-fast variant for the smoke command / quick sanity checks."""
        return replace(
            self, time_points=5, device="cpu", step_size=0.2, log_every=20,
            phi1_layers=[2, 32, 32, 2], phi2_layers=[2, 64, 64, 2],
            steps1=150, steps2=150, steps_base_large=150, steps_base_small=150,
            exp_name="decomp_smoke",
        )

    def to_dict(self) -> dict:
        return asdict(self)
