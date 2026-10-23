import numpy as np
import pytest

from cpslib.attack import plan_fdi
from cpslib.estimation import StateEstimator


@pytest.fixture
def attack_results(network_for, solution_for, layout_for, ns3_noise):
    net = network_for(14)
    voltage = solution_for(14)
    layout = layout_for(14, 2)
    target = 6
    index = net.idx[target]
    fake = abs(voltage[index]) - 0.05
    z = ns3_noise(layout.H @ voltage, np.random.default_rng(0))
    measurements = {key: z[row] for key, row in layout.row.items()}
    baseline = StateEstimator(layout).estimate(measurements)
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
        results[mode] = StateEstimator(layout).estimate(attacked)
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "S11: estimator weights depend on measured |z| (sig = sigma*|z|), "
        "so J is not exactly invariant under a = H c; decision pending in WP2"
    ),
)
def test_stealthy_fdi_preserves_residual_statistic(attack_results):
    baseline, results, _, _ = attack_results
    assert results["stealthy"]["J"] == pytest.approx(baseline["J"], rel=1e-9, abs=1e-12)


def test_simple_fdi_on_noncritical_channel_raises_j(attack_results):
    baseline, results, _, _ = attack_results
    simple = results["simple"]
    assert simple["observable"]
    assert simple["J"] > baseline["J"]
    assert simple["J"] > simple["threshold"]
    assert simple["alarm"]
