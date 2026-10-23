import numpy as np
import pytest

from cpslib.estimation import StateEstimator


@pytest.mark.parametrize("case", [14, 118])
def test_ns3_noise_bdd_calibration(case, solution_for, layout_for, ns3_noise, experiment_configs):
    ns3, config, _, _ = experiment_configs(case)
    layout = layout_for(case, 1)
    z_true = layout.H @ solution_for(case)
    alpha = 0.01
    # Use the effective sigma derived by the default experiment config, so a
    # regression in run_experiment's calibration fails the statistical checks.
    estimator = StateEstimator(layout, sigma=config["se_sigma"], alpha=alpha)
    rng = np.random.default_rng(0)
    ratios, alarms = [], []
    for _ in range(1000):
        z = ns3_noise(z_true, rng, **ns3["noise"])
        result = estimator.estimate({key: z[row] for key, row in layout.row.items()})
        assert result["observable"]
        assert result["dof"] == 2 * (layout.H.shape[0] - layout.net.n)
        ratios.append(result["J"] / result["dof"])
        alarms.append(result["alarm"])
    mean_ratio = float(np.mean(ratios))
    false_alarm_rate = float(np.mean(alarms))
    diagnostic = f"IEEE {case}: mean J/dof={mean_ratio:.6f}, false alarms={false_alarm_rate:.4f}"
    # Both statistics are computed over all trials before either assertion.
    print(diagnostic)
    assert 0.5 * alpha <= false_alarm_rate <= 2 * alpha, diagnostic
    assert 0.9 <= mean_ratio <= 1.1, diagnostic


def generator_q(net, voltage):
    injection = voltage * np.conj(net.Ybus @ voltage) * net.base
    return {bus["id"]: injection[i].imag + bus["qd"] for i, bus in enumerate(net.buses)}


def test_base_case_generator_reactive_limits(network_for, solution_for):
    violations = []
    for case in (14, 30, 39, 57, 118, 300):
        net = network_for(case)
        qg = generator_q(net, solution_for(case))
        reference_buses = {bus["id"] for bus in net.buses if bus["type"] == 3}
        generators = {g["bus"] for g in net.gens if g["status"]} - reference_buses
        for bus in sorted(generators):
            active = [g for g in net.gens if g["status"] and g["bus"] == bus]
            low = sum(g["qmin"] for g in active)
            high = sum(g["qmax"] for g in active)
            if not low - 1e-3 <= qg[bus] <= high + 1e-3:
                violations.append(f"IEEE {case} bus {bus}: {qg[bus]:.6f} MVAr outside [{low}, {high}]")
    # Check every case before failing so a single violation cannot hide others.
    assert not violations, "\n".join(violations)


def test_ieee118_generator49_high_setpoint_respects_reactive_limit(network_for):
    net = network_for(118)
    voltage, _, converged = net.solve_pf(vset={49: 1.12})
    assert converged
    assert generator_q(net, voltage)[49] <= 210 + 1e-3
