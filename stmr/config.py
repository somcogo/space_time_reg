"""Typed experiment configuration.

Replaces the previous 61-flag argparse block and the in-place ``config.*`` mutation
scattered through ``main.py``. A run is fully described by a ``Config``; experiments live
as YAML files under ``configs/`` and override only the fields they care about.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class Config:
    # --- experiment / logging ---
    exp_name: str = "test/test"
    log_path: str = "log"
    seed: int = 0
    device: str = "cuda"
    log_level: str = "info"
    log_cadence: int = 10
    debug: bool = False

    # --- data ---
    dataset: str = "cmr_P001"
    time_points: int = 20
    start_frame: int = 0
    slice_number: int = 0
    factor: int = 4
    mask: str = "st"
    random_mask: bool = False

    # --- initial reconstruction ---
    init: str = "zero"
    init_skip: bool = False
    use_nmapg: bool = False
    recon_epochs: int = 200
    init_lr: float = 0.1
    init_loss: str = "l2"
    init_reg_abs: bool = True
    tol: float = 1e-4
    detach_grads: bool = True
    lambda_init_recon: float = 1.0

    # --- reconstruction regularizer ---
    reg: str = "learned"
    reg_variant: str = "crr"  # 'crr' (weak_convexity=0) or 'wcrr' (weak_convexity=1)
    reg_alpha: float | None = None  # None -> keep the pretrained alpha (do not overwrite)
    recon_scale: float = 0.1  # 0 -> None (keep pretrained scale)

    # --- velocity network ---
    func_name: str = "groupsiren"  # F5: default to the non-stationary ensemble
    siren_depth: int = 3
    siren_dim: int = 256
    siren_omega: int = 30
    last_init_zero: bool = True

    # --- ODE solver ---
    solver: str = "rk4"
    step_size: float = 0.001  # 0 -> adaptive (None)
    atol: float = 1e-9
    rtol: float = 1e-7

    # --- optimisation ---
    lr: float = 0.01
    recon_lr: float = 0.1
    recon_eps: float = 1e-4
    weight_decay: float = 0.1
    epochs: int = 100
    learn_recon: bool = True
    interval: int = 0  # 0 -> epochs (alternation disabled)
    motion_warmup: int = 0
    hard_dc: bool = False
    use_nreps: bool = False  # F2: default off (only supported cmr mode)

    # --- losses ---
    loss: str = "mse"  # F3: MSE for k-space data fidelity
    sim_domain: str = "fourier"  # 'fourier' (k-space DC) or 'image' (image-space DC; equal
    #                              to fourier on fully-sampled data by Parseval, FFT unitary)
    lambda_st: float = 1.0
    lambda_negJ: float = 0.1
    lambda_grd: float = 1.0
    lambda_lap: float = 1.0
    lambda_pgr: float = 1.0
    lambda_hel: float = 1.0
    lambda_recon: float = 1.0
    lambda_rl2: float = 1.0
    lambda_mcdc: float = 0.0
    imdiff_warp_mag: bool = False  # warp magnitude (not complex) in the E3 image loss

    # --- derived / internal (filled by finalize) ---
    schedule: list = field(default_factory=lambda: [1])
    func_kwargs: dict = field(default_factory=dict)

    @classmethod
    def field_names(cls):
        return {f.name for f in fields(cls)}

    @classmethod
    def from_yaml(cls, path):
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        # Coerce to the annotated field type. Guards against the PyYAML gotcha where an
        # unsigned exponent like "1.0e4" is loaded as a str instead of a float.
        coerced = {k: _coerce_to_type(v, known[k].type) for k, v in data.items()}
        return cls(**coerced)

    def finalize(self):
        """Normalise coupled fields once, instead of mutating config all over main.py."""
        if self.recon_scale == 0:
            self.recon_scale = None
        if self.step_size == 0:
            self.step_size = None
        if self.interval == 0:
            self.interval = self.epochs
        return self

    def to_dict(self):
        return asdict(self)


def build_velocity_kwargs(config: Config, dims: int) -> dict:
    """Build the velocity-network constructor kwargs for a given spatial dimensionality."""
    layers = [dims] + config.siren_depth * [config.siren_dim] + [dims]
    kwargs = {
        "layers": layers,
        "omega": config.siren_omega,
        "last_init_zero": config.last_init_zero,
    }
    if "group" in config.func_name:
        kwargs["groups"] = config.time_points - 1
    return kwargs


def parse_cli(argv=None) -> Config:
    """Load a Config from ``--config file.yaml`` plus optional ``--key value`` overrides."""
    parser = argparse.ArgumentParser(description="space-time motion reconstruction")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="override config fields, e.g. --set epochs=5 lr=1e-4",
    )
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config) if args.config else Config()
    for override in args.set:
        key, _, raw = override.partition("=")
        if key not in Config.field_names():
            raise ValueError(f"Unknown override key: {key!r}")
        current = getattr(config, key)
        setattr(config, key, _coerce(raw, current))
    return config.finalize()


def _coerce_to_type(value, type_annotation):
    """Coerce a YAML-loaded value to the dataclass field's declared type."""
    ann = str(type_annotation)
    if value is None:
        # An explicit null (e.g. ``reg_alpha:`` in YAML) stays None regardless of the
        # declared type, so optional fields can be set back to their "keep pretrained"
        # sentinel without tripping float(None).
        return None
    if "bool" in ann:
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)
    if "int" in ann and "float" not in ann:
        return int(float(value))
    if "float" in ann:
        return float(value)
    return value


def _coerce(raw: str, current):
    """Coerce a string override to the type of the current field value."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if current is None:
        # Optional field (e.g. reg_alpha) whose current value is the None sentinel.
        # "none"/"null"/"" set it back to None; anything else is treated as a float.
        if raw.lower() in ("none", "null", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw
