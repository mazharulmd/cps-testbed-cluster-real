# Changelog

## Unreleased

- Add noisy PMU replay, FDI ramps, command drop/delay/forgery, and coordinated command forgery with stealthy FDI to the CLI, scenario schema, API, and both dashboards. Existing attack defaults and NS-3 attack dictionaries are preserved; FDI `ramp_s: 0` retains the step.
- Replay records `[start - record_s, start)` and loops the recorded noisy phasors with current frame indices/timestamps during `[start, end)`. An explicit target selects observing PMUs; null/omitted selects all PMUs. Recording defaults to `min(start, end - start)` (> 0 and <= start). A recording window containing no sampled frames leaves those PMUs unchanged (zero replay count).
- Command attacks require an in-service generator bus, defaulting to `nearest_generators` for the default FDI target. Drop/delay intercept at network ingress inside `[start, end)`; delayed commands continue through the generator link even after the window ends. Forgery sends once at `start`, impersonating the control center through its existing UDP source socket, with `vset` default 1.12 pu (0.8–1.2). Normal generator delivery logs reason `attack: forged command`; the CC setpoint guess remains unchanged (S2, deferred to WP4).
- Coordinated attacks plan stealthy FDI at the operating point after the forged setpoint, displaying the pre-attack target voltage. Earlier AVR/load events are included in that planning point; stochastic load drift and closed-loop changes cannot be predicted by the offline plan. The existing adaptive reference tracks changing voltage when the target PMU is compromised.
- Append replay and command attack counters to NS-3 and experiment summaries, including explicit replay scope and generator targets. Command counters count intercepted/forged attempts, not acknowledgments; `commands_sent` includes intercepted drops and `commands_delivered` remains actual generator receptions. CSV column sets are unchanged.
- Add 14 short builtin-solver demonstrations each for IEEE 14 and IEEE 118, plus pure Python schema/config/API/summary regression coverage. NS-3 compilation and HELICS federation verification require the reviewer environment.
- Fix physically impossible generator reactive output in power flow: enforce in-service PV bus Qmin/Qmax by default (reference buses excluded). The built-in solver switches all violating buses to PQ each round; `solve_pf(qlim=False)`, experiment `--no-qlim`, or scenario/API `qlim: false` restores unconstrained behavior. GridPACK XML uses its integer `qlim` option (`1`/`0`). Existing scenario files default to enforcement. This changes operating points and experiment results.
- Append `q_limited` to `grid_steps.csv`, containing space-separated `bus:max` / `bus:min` entries for QSS power flows. See `documentation/qlim.md` for solver behavior, GridPACK evidence, and validation.
- Fix PMU bad-data detection calibration: derive the experiment's `se_sigma` from
  NS-3 polar noise, changing the effective default from 0.002 to 0.001. NS-3 noise
  defaults remain 0.001 for both magnitude and angle. Configure them with
  `--noise-mag` / `--noise-ang` (scenario `pmu.noise_mag` / `pmu.noise_ang`), or
  override estimation noise with `--se-sigma` (scenario `control.se_sigma`). The
  effective sigma is recorded in configs and `meta.json` / `summary.json` under
  `control_center.se_sigma`. Measured-magnitude estimator weighting is retained.
