## Unreleased

- Fix PMU bad-data detection calibration: derive the experiment's `se_sigma` from
  NS-3 polar noise, changing the effective default from 0.002 to 0.001. NS-3 noise
  defaults remain 0.001 for both magnitude and angle. Configure them with
  `--noise-mag` / `--noise-ang` (scenario `pmu.noise_mag` / `pmu.noise_ang`), or
  override estimation noise with `--se-sigma` (scenario `control.se_sigma`). The
  effective sigma is recorded in configs and `meta.json` / `summary.json` under
  `control_center.se_sigma`. Measured-magnitude estimator weighting is retained.
