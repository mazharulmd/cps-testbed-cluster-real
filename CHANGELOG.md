# Changelog

## Unreleased

- Fix physically impossible generator reactive output in power flow: enforce in-service PV bus Qmin/Qmax by default (reference buses excluded). The built-in solver switches all violating buses to PQ each round; `solve_pf(qlim=False)`, experiment `--no-qlim`, or scenario/API `qlim: false` restores unconstrained behavior. GridPACK XML uses its integer `qlim` option (`1`/`0`). Existing scenario files default to enforcement. This changes operating points and experiment results.
- Append `q_limited` to `grid_steps.csv`, containing space-separated `bus:max` / `bus:min` entries for QSS power flows. See `documentation/qlim.md` for solver behavior, GridPACK evidence, and validation.
- Fix PMU bad-data detection calibration: derive the experiment's `se_sigma` from
  NS-3 polar noise, changing the effective default from 0.002 to 0.001. NS-3 noise
  defaults remain 0.001 for both magnitude and angle. Configure them with
  `--noise-mag` / `--noise-ang` (scenario `pmu.noise_mag` / `pmu.noise_ang`), or
  override estimation noise with `--se-sigma` (scenario `control.se_sigma`). The
  effective sigma is recorded in configs and `meta.json` / `summary.json` under
  `control_center.se_sigma`. Measured-magnitude estimator weighting is retained.
