import numpy as np
import pytest


@pytest.mark.parametrize("case", [14, 118])
@pytest.mark.parametrize("redundancy", [1, 2])
def test_placement_observes_every_bus(case, redundancy, network_for, layout_for):
    net = network_for(case)
    layout = layout_for(case, redundancy)
    counts = np.zeros(net.n, dtype=int)
    for p in range(len(layout.pmus)):
        rows = [row for (pmu, _), row in layout.row.items() if pmu == p]
        # A voltage row sees the PMU bus; its current rows also see far ends.
        # Count a bus once per PMU, even with parallel incident branches.
        block = layout.H[rows].copy()
        block.eliminate_zeros()
        seen = set(block.indices)
        assert seen == layout.sees[p]
        counts[list(seen)] += 1
    assert np.all(counts >= redundancy), dict(zip(net.bus_ids, counts))
