from collections import deque

import pytest

from cpslib.control import nearest_generators


@pytest.mark.parametrize("case", [14, 30, 39, 57, 118, 300])
def test_generator_buses_select_themselves(case, network_for):
    net = network_for(case)
    generators = {g["bus"] for g in net.gens if g["status"]}
    nearest = nearest_generators(net, generators)
    assert set(nearest) == set(net.bus_ids)
    assert set(nearest.values()) <= generators
    for bus in generators:
        assert nearest[bus] == bus


def test_ieee14_bus14_selects_generator_at_minimum_hop_distance(network_for):
    net = network_for(14)
    generators = {g["bus"] for g in net.gens if g["status"]}
    # Bus 14 is a load bus; compute distances independently from topology edges.
    start = 14
    assert start not in generators
    distances = {start: 0}
    queue = deque([start])
    while queue:
        bus = queue.popleft()
        for branch in net.branches:
            if bus not in (branch["from"], branch["to"]):
                continue
            other = branch["to"] if bus == branch["from"] else branch["from"]
            if other not in distances:
                distances[other] = distances[bus] + 1
                queue.append(other)
    chosen = nearest_generators(net, generators)[start]
    assert chosen in generators
    assert distances[chosen] == min(distances[g] for g in generators)
