import numpy as np
import pytest

from cpslib.attack import plan_fdi
from cpslib.estimation import StateEstimator


@pytest.fixture
def attack_results(network_for, solution_for, layout_for, ns3_noise, experiment_configs):
    net = network_for(14)
    voltage = solution_for(14)
    layout = layout_for(14, 2)
    ns3, config, _, _ = experiment_configs(14)
    target = 6
    index = net.idx[target]
    fake = abs(voltage[index]) - 0.05
    z = ns3_noise(layout.H @ voltage, np.random.default_rng(0), **ns3["noise"])
    measurements = {key: z[row] for key, row in layout.row.items()}
    baseline = StateEstimator(layout, sigma=config["se_sigma"]).estimate(measurements)
    assert baseline["observable"] and not baseline["alarm"]
    positions = {pm["bus"]: p for p, pm in enumerate(layout.pmus)}
    results = {}
    for mode in ("stealthy", "simple"):
        plan = plan_fdi(layout, voltage, target, fake, mode)
        attacked = measurements.copy()
        changed_rows = []
        # Apply exactly the per-PMU bus/channel offsets used by check_bdd.py.
        for bus, deltas in plan["deltas"].items():
            p = positions[bus]
            for delta in deltas:
                key = (p, delta["ch"])
                attacked[key] += complex(delta["dre"], delta["dim"])
                changed_rows.append(layout.row[key])
        if mode == "simple":
            assert len(changed_rows) == 1
            remaining = np.delete(layout.H.toarray(), changed_rows, axis=0)
            # Removing the attacked channel leaves full rank: it is non-critical.
            assert np.linalg.matrix_rank(remaining) == net.n
        results[mode] = StateEstimator(layout, sigma=config["se_sigma"]).estimate(attacked)
    expected_shift = (fake - abs(voltage[index])) * np.exp(1j * np.angle(voltage[index]))
    return baseline, results, index, expected_shift


def test_stealthy_fdi_shifts_target_estimate(attack_results):
    baseline, results, index, expected_shift = attack_results
    stealthy = results["stealthy"]
    assert stealthy["observable"] and not stealthy["alarm"]
    # Allow the small effect of reweighting noisy measurements on the estimate.
    np.testing.assert_allclose(
        stealthy["V"][index] - baseline["V"][index], expected_shift, rtol=0, atol=1e-4
    )


def test_stealthy_fdi_preserves_residual_statistic(
    solution_for, layout_for, ns3_noise, experiment_configs
):
    layout = layout_for(14, 2)
    voltage = solution_for(14)
    ns3, config, _, _ = experiment_configs(14)
    alpha = config["bdd_alpha"]
    estimator = StateEstimator(layout, sigma=config["se_sigma"], alpha=alpha)
    target = 6
    plan = plan_fdi(layout, voltage, target, abs(voltage[layout.net.idx[target]]) - 0.05, "stealthy")
    offsets = attack_offsets(layout, plan)
    z_true = layout.H @ voltage
    rng = np.random.default_rng(0)
    alarms, relative_changes = [], []
    for _ in range(300):
        z = ns3_noise(z_true, rng, **ns3["noise"])
        measurements = {key: z[row] for key, row in layout.row.items()}
        baseline = estimator.estimate(measurements)
        attacked = estimator.estimate({key: value + offsets.get(key, 0) for key, value in measurements.items()})
        assert baseline["observable"] and attacked["observable"]
        alarms.append(attacked["alarm"])
        relative_changes.append(abs(attacked["J"] / baseline["J"] - 1))
    rate, mean_change = float(np.mean(alarms)), float(np.mean(relative_changes))
    diagnostic = f"stealthy: alarm rate={rate:.4f}, mean |J_attack/J_clean - 1|={mean_change:.6f}"
    print(diagnostic)
    assert rate <= 2 * alpha, diagnostic
    # Absolute relative changes prevent opposite-sign differences cancelling.
    assert mean_change <= 0.1, diagnostic


def attack_offsets(layout, plan):
    positions = {pm["bus"]: p for p, pm in enumerate(layout.pmus)}
    return {(positions[bus], d["ch"]): complex(d["dre"], d["dim"])
            for bus, deltas in plan["deltas"].items() for d in deltas}


def test_default_noise_detects_small_simple_fdi(solution_for, layout_for, ns3_noise, experiment_configs):
    layout = layout_for(118, 1)
    net = layout.net
    voltage = solution_for(118)
    ns3, config, _, _ = experiment_configs(118)
    estimator = StateEstimator(layout, sigma=config["se_sigma"], alpha=config["bdd_alpha"])
    # Same most-measured bus selection and fixed offset as the reference check.
    target = net.bus_ids[int(np.argmax(np.abs(layout.H).sum(axis=0)))]
    plan = plan_fdi(layout, voltage, target, abs(voltage[net.idx[target]]) - 0.01, "simple")
    offsets = attack_offsets(layout, plan)
    assert len(offsets) == 1
    z_true = layout.H @ voltage
    rng = np.random.default_rng(0)
    alarms = []
    for _ in range(200):
        z = ns3_noise(z_true, rng, **ns3["noise"])
        measurements = {key: z[row] + offsets.get(key, 0) for key, row in layout.row.items()}
        result = estimator.estimate(measurements)
        assert result["observable"]
        alarms.append(result["alarm"])
    rate = float(np.mean(alarms))
    diagnostic = f"IEEE 118 bus {target}: 0.01 pu simple FDI detection={rate:.4f}"
    print(diagnostic)
    assert rate >= 0.9, diagnostic


def test_simple_fdi_on_noncritical_channel_raises_j(attack_results):
    baseline, results, _, _ = attack_results
    simple = results["simple"]
    assert simple["observable"]
    assert simple["J"] > baseline["J"]
    assert simple["J"] > simple["threshold"]
    assert simple["alarm"]
