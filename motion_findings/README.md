# motion_findings

Evidence on whether motion modelling helps **Cartesian CMR cine reconstruction in the CMRxRecon
challenge setting** — i.e. a **static** undersampling mask (same sampled k-space rows in every
frame). Packaged for independent verification and for presenting to a supervisor.

**Start with [SUMMARY.md](SUMMARY.md)** — it states each claim, the numbers, and the exact command
that proves it. Scripts run from the repository root and take optional patient/slice/factor
arguments so the findings can be replicated across cases.

| file | proves | cost |
|---|---|---|
| `verify_1_mask_static.py` | the challenge mask is static across time | seconds |
| `verify_2_static_sharing_useless.py` | with a static mask, motion-free temporal sharing recovers nothing (motion is the only temporal lever) | seconds |
| `verify_3_phase.py` | phase is temporally incoherent (complex motion-compensated coupling is counterproductive) | seconds |
| `verify_4_motion_bug_audit.py` | the motion path is bug-free (correct direction + recovers known motion) | ~1 min (GPU) |
| `run_decomposition.sh` + `summarize_decomposition.py` | how much motion adds over a no-motion refinement, per acceleration | ~10 min/factor (GPU) |

Scope: k-t (time-interleaved) sampling is **out of scope** here — it is an acquisition choice not
available from the challenge's subsampled data (see Claim 0 in SUMMARY.md).
