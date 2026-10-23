import json
from pathlib import Path

import numpy as np
import pytest

import run_experiment


@pytest.mark.parametrize(
    "mag,ang,expected",
    [(0.001, 0.001, 0.001), (0.003, 0.004, np.sqrt(0.0000125)),
     (0, 0.002, 0.002 / np.sqrt(2)), (0, 0, 0)],
)
def test_sigma_from_polar_noise(mag, ang, expected):
    from cpslib.estimation import sigma_from_polar_noise

    assert sigma_from_polar_noise(mag, ang) == pytest.approx(expected)
    assert sigma_from_polar_noise(ang, mag) == pytest.approx(expected)


@pytest.mark.parametrize("mag,ang", [(-0.001, 0.001), (0.001, -0.001), (np.nan, 0), (0, np.inf)])
def test_sigma_from_polar_noise_rejects_invalid_values(mag, ang):
    from cpslib.estimation import sigma_from_polar_noise

    with pytest.raises(ValueError, match="finite and nonnegative"):
        sigma_from_polar_noise(mag, ang)


@pytest.mark.parametrize(
    "options,noise,expected",
    [
        ((), {"mag": 0.001, "ang": 0.001}, 0.001),
        (("--noise-mag", "0.003", "--noise-ang", "0.004"),
         {"mag": 0.003, "ang": 0.004}, np.sqrt(0.0000125)),
        (("--noise-mag", "0", "--noise-ang", "0"), {"mag": 0, "ang": 0}, 0),
        (("--noise-mag", "0.003", "--noise-ang", "0.004", "--se-sigma", "0.007"),
         {"mag": 0.003, "ang": 0.004}, 0.007),
    ],
)
def test_experiment_noise_configs_and_summary(options, noise, expected, experiment_configs, monkeypatch):
    ns3, config, meta, run_dir = experiment_configs(14, options)
    assert ns3["noise"] == noise
    assert config["se_sigma"] == pytest.approx(expected)
    assert meta["control_center"]["se_sigma"] == pytest.approx(expected)
    assert json.loads((run_dir / "meta.json").read_text())["control_center"]["se_sigma"] == pytest.approx(expected)
    if not options:
        assert meta["args"]["se_sigma"] is None  # metadata distinguishes derivation from override

    # Verify the CC config produced by orchestration without launching federates.
    class FinishedProcess:
        returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(run_experiment.cluster, "launch", lambda *args: FinishedProcess())
    monkeypatch.setattr(run_experiment.time, "sleep", lambda _: None)
    cl = {"head": "localhost", "placement": {}, "nodes": [], "mpi": {}}
    codes, wall = run_experiment.run_federation(str(run_dir), cl, 23500)
    cc = json.loads((run_dir / "cc_config.json").read_text())
    assert cc["se_sigma"] == pytest.approx(expected)

    # Minimal result streams exercise the real summary writer, including an
    # explicit override whose effective value differs from the noise derivation.
    (run_dir / "ns3_summary.json").write_text("{}")
    meta["args"]["grid_step"] = 0.5
    monkeypatch.setattr(run_experiment, "read_csv", lambda path:
                        [{"t": "0", "v1": "1"}] if Path(path).name == "grid_truth.csv" else [])
    summary = run_experiment.analyze(str(run_dir), meta, codes, wall)
    assert summary["control_center"]["se_sigma"] == pytest.approx(expected)
    assert json.loads((run_dir / "summary.json").read_text())["control_center"]["se_sigma"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "option,value",
    [("--noise-mag", "-0.001"), ("--noise-ang", "0.051"), ("--noise-mag", "nan"),
     ("--noise-ang", "inf"), ("--se-sigma", "0"), ("--se-sigma", "-0.001"),
     ("--se-sigma", "nan"), ("--se-sigma", "inf")],
)
def test_cli_rejects_invalid_noise(option, value):
    with pytest.raises(SystemExit) as exc:
        run_experiment.parse_args([option, value])
    assert exc.value.code == 2
