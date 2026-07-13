# Space-time reconstruction (stmr)

Motion-regularised reconstruction of dynamic (cine) MRI from undersampled k-space.
A strong per-frame prior is built with a learned convex-ridge regulariser (nmAPG init),
then the images and a SIREN-based velocity field are optimised jointly so that temporal
consistency (an ODE flow / deformation) fills in the missing k-space. The theory is in
[main.tex](main.tex).

## Install

```bash
pip install -e ".[dev]"      # runtime + dev (pytest, ruff)
```

The two pretrained CRR/WCRR regulariser weights live under `weights/bilevel_CT/`.

## Run

Experiments are plain YAML configs under `configs/`:

```bash
stmr-run --config configs/cmr_soft_con.yaml
# override any field ad hoc:
stmr-run --config configs/cmr_soft_con.yaml --set epochs=5 recon_epochs=5
```

`configs/cmr_soft_con.yaml` is the full pipeline (k-t mask + hard DC + WCRR) on
CMRxRecon P001.

## Layout

```
stmr/
  config.py        typed Config + YAML loader (replaces the old argparse/bash sprawl)
  cli.py           `stmr-run` entrypoint
  pipeline.py      data -> init recon -> registration -> eval -> save
  data/            k-space loading, forward operators (FFT + mask), init strategies
  recon/ ...       nmapg.py, recon_init.py: initial per-frame reconstruction
  losses/          data fidelity, CRR/WCRR regulariser, image consistency, motion reg
  models/          Siren + GroupedSiren velocity nets (+ registry factory)
  metrics/, utils/, regularizers/
configs/           experiment YAMLs
tests/             pytest suite (fft, masks, losses, ode, config, smoke)
weights/           pretrained regulariser weights
```

## Develop

```bash
ruff check .
pytest
```
