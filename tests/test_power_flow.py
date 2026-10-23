import csv

import numpy as np
import pytest


@pytest.mark.parametrize("case", [14, 30, 39, 57, 118, 300])
def test_power_flow_matches_reference(case, repo_root, network_for, solution_for):
    net = network_for(case)
    voltage = solution_for(case)
    with (repo_root / "cases" / f"ieee{case}" / "reference.csv").open() as stream:
        reference = {int(row["bus_id"]): row for row in csv.DictReader(stream)}
    assert set(reference) == set(net.bus_ids)
    vm = np.array([float(reference[b]["vm_pu"]) for b in net.bus_ids])
    va = np.array([float(reference[b]["va_deg"]) for b in net.bus_ids])
    np.testing.assert_allclose(np.abs(voltage), vm, rtol=0, atol=1e-5)
    # Compare angles modulo a full turn, preserving the case's reference angle.
    error_deg = np.rad2deg(np.angle(voltage * np.exp(-1j * np.deg2rad(va))))
    np.testing.assert_allclose(error_deg, 0, rtol=0, atol=1e-3)
