# Limit v0.1 symmetry support to C1 and I

Version 0.1 accepts only no symmetry (`C1`) and CryoSPARC's documented icosahedral convention (`I`). It rejects `I1`, `I2`, `C<n>`, `D<n>`, `T`, and `O` instead of silently mapping them to an unverified operator embedding. Symmetry remains part of pose folding and Representative Symmetry View selection; only the human-facing nearest-axis label and distance are removed. `C<n>`, `D<n>`, and `T` are deferred until CryoSPARC convention integration tests exist. See `docs/research/cryosparc-icosahedral-conventions.md`.
