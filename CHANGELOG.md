# Changelog

## Unreleased

- Fix physically impossible generator reactive output in power flow: enforce in-service PV bus Qmin/Qmax by default (reference buses excluded). The built-in solver switches all violating buses to PQ each round; `solve_pf(qlim=False)`, experiment `--no-qlim`, or scenario/API `qlim: false` restores unconstrained behavior. GridPACK XML uses its integer `qlim` option (`1`/`0`). Existing scenario files default to enforcement. This changes operating points and experiment results.
- Append `q_limited` to `grid_steps.csv`, containing space-separated `bus:max` / `bus:min` entries for QSS power flows. See `documentation/qlim.md` for solver behavior, GridPACK evidence, and validation.
