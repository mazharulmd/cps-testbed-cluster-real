Run from the repository root with Python 3.12 and numpy, scipy, pyyaml,
pydantic (v2), and pytest available:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
# Supplied development environment:
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B -m pytest -q
```

No HELICS, GridPACK, MPI, or NS-3 is required. Optional pandapower cross-checks
skip if it is absent. See `documentation/qlim.md` for reactive-limit validation. Seeded statistical
checks reproduce NS-3's independent polar noise draws and use the effective
sigma from the default experiment config (derived from the configured magnitude
and angle noise). F1 calibration checks use 1000 trials each on IEEE 14 and 118
with minimum PMU placement: false alarms must fall within 0.5–2 times alpha=0.01
and mean J/dof within [0.9, 1.1]. A 200-trial IEEE 118 test requires at least 90%
detection of a 0.01 pu simple FDI on the most-measured bus.

S11 retains measured-|z| weighting (`sig = sigma*|z|`); exact J invariance under
`a = H c` is not expected. On IEEE 14 with redundancy 2, target bus 6 and a
−0.05 pu stealthy FDI, 300 paired noise draws must give an alarm rate at most
2*alpha and mean absolute relative J change at most 10%. F1, F2 and S11 are passing
regressions (F2: generator reactive limits, WP3).

Noise config tests cover derivation, explicit overrides, persisted effective
sigma, CLI validation, and scenario → API model → CLI forwarding. The API model
is loaded from its AST to avoid FastAPI and background-worker dependencies.

Passing estimator checks cover complex-valued degrees of freedom and chi-square
thresholds on IEEE 14 and 118, including excluded channels. PMU-loss checks use
IEEE 14 at redundancies 1 and 2. IEEE 14's minimum placement has 16 channels and
every PMU removal leaves fewer than 14, so a further minimum-placement check on
IEEE 118 selects a PMU from `layout.sees` whose removal leaves a bus unseen while
retaining at least 118 channels. It also checks that only the pseudo-measurement
factorization runs: otherwise a singular-matrix fallback could mask removal of
the topological observability guard. Suppressing alarms when unobservable records
current behaviour; the control-center fix comes in WP4.
