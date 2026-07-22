"""Two staged ablation ladders.

Both ladders start at "fully-sampled GT images, registration only, no k-space
undersampling" (S0) and end at the same full pipeline (matching
``configs/cmr_soft_con.yaml`` at time_points=6): free recon, Fourier-space sim loss,
real nmAPG init, k-space undersampling + hard DC, and the learned WCRR prior. They differ
in the ORDER those four components are switched on, which isolates whether a given
component's effect on PSNR depends on what else is already active when it's introduced:

  ladder 1: S0 -> + sim loss/recon learning -> + FT -> + nmAPG init -> + undersampling
            -> + learned reg
  ladder 2: S0 -> + sim loss/recon learning -> + learned reg -> + nmAPG init -> + FT
            -> + undersampling

Stages are expressed as ``BASE`` (constants held fixed across both ladders) + a per-stage
``overrides`` dict, materialised into a ``Config`` by :func:`stmr.ablation.runner.build_config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# Knobs held constant across both ladders so that only the intended component changes per
# stage. Data/geometry (T, patient slice), the velocity net, the ODE solver and the base
# optimiser lrs all match configs/cmr_soft_con.yaml; imdiff_warp_mag is left at the Config
# default (matching the yaml).
BASE = dict(
    device="cuda",
    seed=0,
    time_points=6,
    start_frame=0,
    slice_number=0,
    factor=4,
    mask="st",
    func_name="groupsiren",
    siren_depth=3,
    siren_dim=64,
    last_init_zero=True,
    solver="euler",
    step_size=0.1,
    atol=1e-8,
    rtol=1e-6,
    lr=1e-4,
    recon_lr=1e-5,
    recon_eps=1e-4,
    weight_decay=0.0,
    interval=0,
    motion_warmup=0,
    loss="mse",
    lambda_grad_phi=1e-2,
    lambda_grd=0.0,
    lambda_lap=0.0,
    lambda_hel=0.0,
    lambda_mcdc=0.0,
    log_cadence=50,
    debug=False,
)


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    overrides: dict
    primary_metric: str
    criterion: str
    expectation: str
    # (this_row, prev_row) -> bool. None means "report only" (no pass/fail gate). prev_row
    # is None for the first stage in a ladder.
    pass_fn: Optional[Callable[[dict, Optional[dict]], bool]] = None
    # If set, load Config.from_yaml(yaml) as the base instead of BASE, then apply overrides.
    yaml: Optional[str] = None


def _merge(*dicts: dict) -> dict:
    out = {}
    for d in dicts:
        out.update(d)
    return out


# --- pass/fail predicates -------------------------------------------------------------
# Thresholds on the anchoring stages are provisional: treat the first real (500-epoch) run
# as calibration and tighten them to the observed anchored values.

def _s0_pass(row, _prev):
    return row["imdiff_warp"] <= 0.7 * row["imdiff_base"] and (row["vel_rms"] or 0) > 0


def _anchor_pass(row, _prev):
    # Fully-sampled data consistency (image- or Fourier-space) must anchor the recon at GT.
    return row["full_psnr"] >= 40.0


def _fft_equiv_pass(row, prev):
    # The FFT works as intended: switching the sim loss from image space to Fourier space
    # must reproduce the same anchored PSNR (Parseval). Also still anchors at GT.
    return (prev is not None and row["full_psnr"] >= 40.0
            and abs(row["full_psnr"] - prev["full_psnr"]) <= 2.0)


def _doesnt_regress_pass(row, prev):
    # Generic "this change shouldn't cost much PSNR" gate, used whenever a new component
    # is switched on while the problem is still fully-sampled (so it's still well-posed).
    return prev is not None and (prev["full_psnr"] - row["full_psnr"]) <= 2.0


def _improves_pass(row, prev):
    return prev is not None and row["full_psnr"] >= prev["full_psnr"]


# --- shared building blocks -------------------------------------------------------------
# Each block is one component's worth of overrides; stages are built by cumulatively
# merging the blocks introduced so far, so the two ladders can reuse the exact same blocks
# in different orders.

_S0_OVERRIDES = dict(
    dataset="cmr_test2", init="gt", init_skip=True, learn_recon=False,
    hard_dc=False, lambda_st=0.0, lambda_recon=0.0, lambda_rl2=1e4,
)

# + sim loss, recon learning: free (jointly optimised) recon starting from GT, fully
# sampled, image-space data consistency.
_SIM_RECON = dict(
    dataset="cmr_test2", init="gt", init_skip=True, learn_recon=True,
    hard_dc=False, lambda_st=1.0, lambda_recon=0.0, lambda_rl2=1e4,
    sim_domain="image",
)

# + FT: move the sim loss from image space to Fourier space (FFT data-consistency check).
_FT = dict(sim_domain="fourier")

# + real nmAPG init: replace the GT init with the actual nmAPG-based initial reconstruction
# pathway used by the real pipeline (still runs whatever dataset/domain is active so far).
_NMAPG_INIT = dict(
    init="adj", init_skip=False, use_nmapg=True, recon_epochs=150,
    init_lr=1e-2, init_loss="l2", lambda_init_recon=2e-2,
)

# + undersampling: switch to the real (factor=4) undersampled k-space and turn on hard DC.
_UNDERSAMPLING = dict(dataset="cmr_P001", hard_dc=True)

# + learned reg: turn on the pretrained WCRR prior as a joint-optimisation regularizer.
_LEARNED_REG = dict(
    lambda_recon=1.0, reg="learned", reg_variant="wcrr",
    reg_alpha=1.0, recon_scale=6.0,
)


S0 = StageSpec(
    id="S0",
    title="registration only (k-space fidelity off, recon frozen at GT)",
    overrides=_S0_OVERRIDES,
    primary_metric="imdiff_warp",
    criterion="PASS if imdiff_warp <= 0.7 * imdiff_base and vel_rms > 0",
    expectation="warp(I_t) should match I_{t+1} clearly better than identity",
    pass_fn=_s0_pass,
)


LADDER_1 = [
    S0,
    StageSpec(
        id="S1",
        title="+ sim loss, recon learning (image-space, fully-sampled, GT init)",
        overrides=_merge(_SIM_RECON),
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB (calibrate on first run)",
        expectation="image-space data consistency anchors the recon at GT; reference for "
                    "the FFT check in S2",
        pass_fn=_anchor_pass,
    ),
    StageSpec(
        id="S2",
        title="+ FT (Fourier-space sim loss)",
        overrides=_merge(_SIM_RECON, _FT),
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB and |S2 - S1| <= 2 dB",
        expectation="only switch vs S1 is moving the sim loss to Fourier space; if the FFT "
                    "path is correct the PSNR matches S1 (Parseval)",
        pass_fn=_fft_equiv_pass,
    ),
    StageSpec(
        id="S3",
        title="+ real nmAPG init (still fully-sampled)",
        overrides=_merge(_SIM_RECON, _FT, _NMAPG_INIT),
        primary_metric="full_psnr",
        criterion="PASS if S2.full_psnr - S3.full_psnr <= 2 dB",
        expectation="swapping the GT init for the real nmAPG init pathway shouldn't cost "
                    "PSNR while the problem is still fully-sampled",
        pass_fn=_doesnt_regress_pass,
    ),
    StageSpec(
        id="S4",
        title="+ undersampling + hard DC",
        overrides=_merge(_SIM_RECON, _FT, _NMAPG_INIT, _UNDERSAMPLING),
        primary_metric="full_psnr",
        criterion="report-only: expect d_vs_init < 0 (drift off GT)",
        expectation="SUSPECTED BREAK -- under subsampled DC the objective minimum is away "
                    "from GT, so PSNR should drop here",
        pass_fn=None,
    ),
    StageSpec(
        id="S5",
        title="+ learned (WCRR) reg [full pipeline]",
        overrides=_merge(_SIM_RECON, _FT, _NMAPG_INIT, _UNDERSAMPLING, _LEARNED_REG),
        primary_metric="full_psnr",
        criterion="PASS if S5.full_psnr >= S4.full_psnr",
        expectation="the learned prior should recover some of the PSNR lost to undersampling",
        pass_fn=_improves_pass,
    ),
]


LADDER_2 = [
    S0,
    StageSpec(
        id="S1",
        title="+ sim loss, recon learning (image-space, fully-sampled, GT init)",
        overrides=_merge(_SIM_RECON),
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB (calibrate on first run)",
        expectation="image-space data consistency anchors the recon at GT",
        pass_fn=_anchor_pass,
    ),
    StageSpec(
        id="S2",
        title="+ learned (WCRR) reg",
        overrides=_merge(_SIM_RECON, _LEARNED_REG),
        primary_metric="full_psnr",
        criterion="PASS if S1.full_psnr - S2.full_psnr <= 2 dB",
        expectation="quantifies how much the learned prior biases the solution at GT, "
                    "before anything else is turned on",
        pass_fn=_doesnt_regress_pass,
    ),
    StageSpec(
        id="S3",
        title="+ real nmAPG init (still fully-sampled)",
        overrides=_merge(_SIM_RECON, _LEARNED_REG, _NMAPG_INIT),
        primary_metric="full_psnr",
        criterion="PASS if S2.full_psnr - S3.full_psnr <= 2 dB",
        expectation="swapping the GT init for the real nmAPG init pathway shouldn't cost "
                    "PSNR while the problem is still fully-sampled",
        pass_fn=_doesnt_regress_pass,
    ),
    StageSpec(
        id="S4",
        title="+ FT (Fourier-space sim loss)",
        overrides=_merge(_SIM_RECON, _LEARNED_REG, _NMAPG_INIT, _FT),
        primary_metric="full_psnr",
        criterion="PASS if full_psnr >= 40 dB and |S4 - S3| <= 2 dB",
        expectation="only switch vs S3 is moving the sim loss to Fourier space; if the FFT "
                    "path is correct the PSNR matches S3 (Parseval)",
        pass_fn=_fft_equiv_pass,
    ),
    StageSpec(
        id="S5",
        title="+ undersampling + hard DC [full pipeline]",
        overrides=_merge(_SIM_RECON, _LEARNED_REG, _NMAPG_INIT, _FT, _UNDERSAMPLING),
        primary_metric="full_psnr",
        criterion="report-only: expect d_vs_init < 0 (drift off GT)",
        expectation="SUSPECTED BREAK -- under subsampled DC the objective minimum is away "
                    "from GT, so PSNR should drop here",
        pass_fn=None,
    ),
]


LADDERS = {"1": LADDER_1, "2": LADDER_2}


def resolve_stage_ids(spec_str: str, stages: list[StageSpec]) -> list[str]:
    """Parse a --stages argument: a range 'S0-S5', a comma list 'S0,S3,S5', or 'all'."""
    stage_by_id = {s.id: s for s in stages}
    spec_str = spec_str.strip()
    if not spec_str or spec_str.lower() == "all":
        return [s.id for s in stages]
    if "-" in spec_str and "," not in spec_str:
        lo, hi = spec_str.split("-")
        order = [s.id for s in stages]
        i, j = order.index(lo.strip()), order.index(hi.strip())
        return order[i:j + 1]
    ids = [tok.strip() for tok in spec_str.split(",") if tok.strip()]
    for sid in ids:
        if sid not in stage_by_id:
            raise ValueError(f"Unknown stage {sid!r}; known: {list(stage_by_id)}")
    return ids
