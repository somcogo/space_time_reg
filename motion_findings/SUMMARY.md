# Does motion modelling help Cartesian CMR cine reconstruction? (static-mask / challenge setting)

**Scope.** The CMRxRecon challenge provides undersampled k-space with a **static** Cartesian mask —
the *same* phase-encode rows are sampled in every cardiac frame. This document asks, strictly in that
setting, whether the space-time method's **motion model** improves reconstruction over an otherwise
identical pipeline without motion. Time-interleaved (k-t) sampling is **out of scope**: it is an
acquisition choice not available from the challenge's subsampled data (Claim 0).

**Setup.** Per-frame images `I_t` reconstructed jointly with a velocity field (grouped SIREN + neural
ODE, adjoint backprop) that warps frames onto each other, plus a learned convex-ridge spatial prior
(CRR) and hard data consistency. Reference experiment: patient P001, slice 0, `T=6` frames, static
mask, acceleration 4/8/10.

**Bottom line.** In the static-mask challenge setting, motion-free temporal sharing is *worthless*
(every frame misses the same rows), so motion is the only mechanism that could exploit the other
frames — and it does **not**: motion adds **≈+0.01 dB** at every acceleration (4/8/10×), and
increasing the motion weight makes it *worse* (−0.04 dB at 3×). Crucially this is not because motion
failed to train: the learned velocity is non-trivial (comparable magnitude to any other run) and
correctly reduces the frame-to-frame residual by ~90% — the per-frame reconstructions are simply
already so similar (~3×10⁻⁷ MSE apart) that warping them onto each other adds essentially nothing to
PSNR. The whole space-time apparatus (CRR refinement + motion + hard DC) adds ~0.08 dB over the
per-frame nmAPG init, of which motion is ~0.01.

> **Metric note.** PSNR/SSIM/NMSE follow the official CMRxRecon evaluation (data range = `gt.max()`,
> no per-image self-normalization; a metric bug was fixed 2026-07-10). Numbers here are internally
> consistent but ~4 dB higher than logs from before that date.

---

## Claim 0 — The challenge mask is static across time

The provided `acc_04` k-space samples the **same rows in every frame** (equispaced every 4th row +
24-row ACS centre; 69/204 rows/frame, verified identical across all 12 frames). Consequently rows
missing in one frame are missing in *all* frames.

**Verify:** `python motion_findings/verify_1_mask_static.py` → "MASK IDENTICAL ACROSS ALL 12 FRAMES? -> True"

---

## Claim 1 — With a static mask, motion-free temporal sharing recovers nothing

Because every frame is missing the same rows, copying or averaging k-space across frames cannot fill
any missing row. A "temporal sliding-window" reconstruction is therefore **identical** to the
zero-filled reconstruction:

| method | PSNR | SSIM |
|---|---:|---:|
| zero-filled | 31.63 | 0.848 |
| temporal sliding window (nearest frame) | 31.63 | 0.848 |

So in the challenge setting the only way to use the other frames at all is **motion-compensated**
reconstruction. This motivates the method — and sets up the question in Claim 2.

**Verify:** `python motion_findings/verify_2_static_sharing_useless.py` → "IDENTICAL to zero-fill".

---

## Claim 2 — Motion adds essentially nothing (≈+0.01 dB); pushing it harder hurts

Decomposition at matched initialization (per-frame nmAPG+CRR reconstruction from static-masked
k-space), 2000 epochs, P001 slice 0. Δmotion = PSNR(with motion) − PSNR(no-motion refinement).

| acceleration | nmAPG+CRR init | no-motion refine | + motion | **Δ motion** | crop Δ |
|---|---:|---:|---:|---:|---:|
| 4× | 31.84 | 31.91 | 31.92 | **+0.01** | +0.05 |
| 8× | 30.96 | 31.02 | 31.03 | **+0.01** | +0.01 |
| 10× | 30.62 | 30.66 | 30.67 | **+0.01** | +0.02 |

Giving motion more weight does not help and slightly hurts (factor 4): motion weight ×3 → 31.87
(**−0.04**); adding a magnitude motion-compensated data-consistency term → 31.90 (−0.01).

The learned motion is genuinely active — velocity magnitude 920–1408 (comparable to any run) and it
reduces the real frame-to-frame residual by 85–92% — so the null result is "motion works but is
useless here," not "motion failed to train." Also note (a) the nmAPG+CRR init (31.84) is only ~0.2 dB
above zero-fill (31.63) — a static mask leaves little for a single-frame prior either; (b) the
no-motion main loop adds only ~0.07 dB, because the init already applied CRR.

**Interpretation (coherent aliasing).** Under a static mask the aliasing is *coherent and identical
in every frame*, so the temporal-consistency term is nearly minimized by zero motion and the motion
that is learned has little corrective value; consecutive frames are already ~3×10⁻⁷ MSE apart. At
higher acceleration the motion is also estimated from more heavily aliased images (chicken-and-egg),
so there is no regime among 4/8/10× where motion becomes useful.

**Verify:** `bash motion_findings/run_decomposition.sh 4` (then 8, 10) →
`python motion_findings/summarize_decomposition.py`.

---

## Claim 3 — The small gain is NOT a bug: the motion path is correct

Two independent, setting-independent checks:

- **(A) Real trained motion, correct direction.** Warping frame *i* with the learned deformation
  reduces its residual to frame *i+1* by **37–98 %** across all saved runs (never increases it → no
  sign/coordinate/convention error).
- **(B) Synthetic recovery.** Known smooth deformations of 1, 3, 6 px fed through the real
  `GroupedSIREN → odeint → grid_sample` stack are recovered to **~100 %** residual reduction.

The motion machinery works; it simply has little to correct once the frames are individually
reconstructed and are already similar.

**Verify:** `python motion_findings/verify_4_motion_bug_audit.py` → all "REDUCES (ok)" / "RECOVERED".

---

## Claim 3b (supporting) — Complex temporal coupling fights the data

Predicting frame *t+1* from frame *t*: copying the **complex** value gives 5.8 % relative residual,
copying only the **magnitude** gives 0.8 %. The phase is temporally incoherent, so any complex-valued
motion-compensated data-consistency term is counterproductive; a magnitude-routed version was tried
and does not change the conclusion.

**Verify:** `python motion_findings/verify_3_phase.py`.

---

## Limitations / what a skeptic should check next

1. **Single case.** All numbers are P001, slice 0, `T=6`. **Before presenting as general, replicate
   the decomposition across several patients/slices** — every script takes patient/slice/factor
   arguments and `run_decomposition.sh` takes a dataset (e.g. `cmr_P002`). Most important robustness
   check.
2. **Anchor, not comparison.** CMRxRecon-2023 winner PromptMR (supervised, multi-coil, 120 training
   subjects) reaches ~39 dB / 0.96 SSIM cine SAX 4×. Not directly comparable — cited only for scale.
   It shows the achievable ceiling comes from a strong *learned spatial/coil* model, not from motion.
3. The coherent-vs-incoherent-aliasing interpretation is a hypothesis; a non-Cartesian dataset would
   be needed to test whether motion helps when the per-frame artifact is incoherent.

## Reproduce everything

```
python motion_findings/verify_1_mask_static.py             # Claim 0  (seconds)
python motion_findings/verify_2_static_sharing_useless.py  # Claim 1  (seconds)
python motion_findings/verify_3_phase.py                   # Claim 3b (seconds)
python motion_findings/verify_4_motion_bug_audit.py        # Claim 3  (GPU, ~1 min)
bash   motion_findings/run_decomposition.sh 4              # Claim 2  (GPU, ~2 runs)
bash   motion_findings/run_decomposition.sh 8
bash   motion_findings/run_decomposition.sh 10
python motion_findings/summarize_decomposition.py          # prints the Δmotion table
```
