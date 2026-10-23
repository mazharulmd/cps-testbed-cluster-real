Run from the repository root with Python 3.12 and numpy, scipy, pyyaml,
pydantic (v2), and pytest available:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
# Supplied development environment:
PYTHONDONTWRITEBYTECODE=1 ~/.boss-venv/bin/python -B -m pytest -q
```

No HELICS, GridPACK, MPI, or NS-3 is required. Optional pandapower cross-checks
skip if it is absent. See `documentation/qlim.md` for reactive-limit validation.
Seeded statistical checks use 300 trials per grid. Strict xfails record F1 (BDD calibration,
WP2) and S11: estimator weights depend on
measured |z| (sig = sigma*|z|), so J is not exactly invariant under a = H c;
decision pending in WP2. S11 is not a confirmed bug. An unexpected pass fails
the suite so the corresponding marker must be reviewed when behaviour changes.

Passing estimator checks cover complex-valued degrees of freedom and chi-square
thresholds on IEEE 14 and 118, including excluded channels. PMU-loss checks use
IEEE 14 at redundancies 1 and 2. IEEE 14's minimum placement has 16 channels and
every PMU removal leaves fewer than 14, so a further minimum-placement check on
IEEE 118 selects a PMU from `layout.sees` whose removal leaves a bus unseen while
retaining at least 118 channels. It also checks that only the pseudo-measurement
factorization runs: otherwise a singular-matrix fallback could mask removal of
the topological observability guard. Suppressing alarms when unobservable records
current behaviour; the control-center fix comes in WP4.
