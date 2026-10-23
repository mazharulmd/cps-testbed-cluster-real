"""Pure Python model fixtures; no federate or simulation runtime is imported."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ("federates", "webapp"):
    sys.path.insert(0, str(ROOT / directory))

from cpslib.network import Network
from cpslib.placement import optimal_placement
from cpslib.pmu import PmuLayout


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def network_for():
    @lru_cache(maxsize=None)
    def load(case):
        return Network(str(ROOT / "cases" / f"ieee{case}" / "topology.json"))

    return load


@pytest.fixture(scope="session")
def solution_for(network_for):
    @lru_cache(maxsize=None)
    def solve(case):
        voltage, _, converged = network_for(case).solve_pf()
        assert converged, f"IEEE {case} power flow did not converge"
        return voltage

    return solve


@pytest.fixture(scope="session")
def layout_for(network_for):
    @lru_cache(maxsize=None)
    def layout(case, redundancy=1):
        net = network_for(case)
        buses, _ = optimal_placement(net, redundancy)
        assert len(buses) == len(set(buses))
        return PmuLayout(net, buses)

    return layout


@pytest.fixture
def ns3_noise():
    def sample(z, rng):
        # NS-3 Sample(): independent normal draws for magnitude and phase.
        return np.abs(z) * (1 + 0.001 * rng.standard_normal(z.size)) * np.exp(
            1j * (np.angle(z) + 0.001 * rng.standard_normal(z.size))
        )

    return sample
