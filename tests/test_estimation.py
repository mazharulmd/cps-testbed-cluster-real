from unittest.mock import Mock

import numpy as np
import pytest
from scipy.stats import chi2

import cpslib.estimation as estimation
from cpslib.estimation import StateEstimator


@pytest.mark.parametrize("case", [14, 118])
@pytest.mark.parametrize("redundancy", [1, 2])
def test_noise_free_state_recovery(case, redundancy, solution_for, layout_for):
    voltage = solution_for(case)
    layout = layout_for(case, redundancy)
    z = layout.H @ voltage
    result = StateEstimator(layout).estimate({key: z[row] for key, row in layout.row.items()})
    assert result["observable"]
    assert result["m"] == layout.H.shape[0]
    np.testing.assert_allclose(result["V"], voltage, rtol=0, atol=1e-8)
    assert result["J"] < 1e-12
    assert not result["alarm"]


@pytest.mark.parametrize("case", [14, 118])
@pytest.mark.parametrize("excluded_count", [0, 2])
def test_observable_dof_and_threshold(case, excluded_count, solution_for, layout_for):
    layout = layout_for(case, 2)
    z = layout.H @ solution_for(case)
    measurements = {key: z[row] for key, row in layout.row.items()}
    # Leave each PMU's current channels available when excluding voltage channels.
    excluded = tuple(key for key in measurements if key[1] == 0)[:excluded_count]
    assert len(excluded) == excluded_count
    alpha = 0.025
    result = StateEstimator(layout, alpha=alpha).estimate(measurements, exclude=excluded)

    m = layout.H.shape[0] - excluded_count
    assert result["observable"]
    assert result["m"] == m
    assert m > layout.net.n
    dof = 2 * (m - layout.net.n)
    assert result["dof"] == dof
    assert result["threshold"] == pytest.approx(chi2.ppf(1 - alpha, dof))


@pytest.mark.parametrize("case", [14, 118])
def test_minimum_placement_loses_observability_without_one_pmu(
    case, solution_for, layout_for, monkeypatch
):
    layout = layout_for(case, 1)
    all_buses = set(range(layout.net.n))
    candidates = []
    for p in range(len(layout.pmus)):
        seen = set().union(*(buses for q, buses in enumerate(layout.sees) if q != p))
        remaining_m = sum(key[0] != p for key in layout.row)
        if seen != all_buses and (case == 14 or remaining_m >= layout.net.n):
            candidates.append(p)
    assert candidates, f"IEEE {case}: no qualifying PMU removal"
    p = candidates[0]
    z = layout.H @ solution_for(case)
    measurements = {key: z[row] for key, row in layout.row.items() if key[0] != p}
    if case == 118:
        # IEEE 14 drops below n for every PMU removal; IEEE 118 isolates the seen check.
        assert len(measurements) >= layout.net.n

    # A singular-matrix failure can hide a missing topological guard. Record solves
    # to verify that an unseen bus goes directly to the pseudo-measurement solve.
    factorize = Mock(wraps=estimation.splu)
    monkeypatch.setattr(estimation, "splu", factorize)
    result = StateEstimator(layout).estimate(measurements)

    assert result["m"] == len(measurements)
    assert result["observable"] is False
    # Current estimator behaviour suppresses alarms when unobservable; the
    # control-center fix comes in WP4.
    assert result["alarm"] is False
    assert factorize.call_count == 1


def test_redundant_placement_stays_observable_without_one_pmu(solution_for, layout_for):
    layout = layout_for(14, 2)
    z = layout.H @ solution_for(14)
    # Redundancy 2 must tolerate losing any one PMU, including all its channels.
    for p in range(len(layout.pmus)):
        seen = set().union(*(buses for q, buses in enumerate(layout.sees) if q != p))
        assert seen == set(range(layout.net.n))
        measurements = {key: z[row] for key, row in layout.row.items() if key[0] != p}
        result = StateEstimator(layout).estimate(measurements)
        assert result["m"] == len(measurements)
        assert result["m"] >= layout.net.n
        assert result["observable"] is True
