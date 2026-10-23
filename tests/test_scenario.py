import pytest

from scenario import Scenario, parse


@pytest.mark.parametrize(
    "path,count", [("webapp/scenario_template.yaml", 1), ("scenarios/demo-ieee14.yaml", 3)]
)
def test_shipped_scenarios_parse(path, count, repo_root):
    scenarios = parse((repo_root / path).read_text())
    assert len(scenarios) == count
    assert all(isinstance(item, Scenario) for item in scenarios)


@pytest.mark.parametrize(
    "text,error",
    [
        ("network: [", "not valid YAML/JSON"),
        ("just a scalar", "mapping"),
        ("- ieee14\n- ieee118", "mapping"),
        ("unexpected: true", "unexpected"),
        ("network: {unexpected: true}", "network.unexpected"),
        ("duration: 1000", "duration"),
        ("network: {loss: 0.95}", "network.loss"),
        ("scenarios: []", "non-empty list"),
    ],
)
def test_invalid_scenarios_raise_value_error(text, error):
    with pytest.raises(ValueError, match=error):
        parse(text)


def test_scenario_list_merges_nested_defaults():
    first, second = parse("""
grid: ieee118
network: {latency_ms: 25, loss: 0.1}
scenarios:
  - name: first
    network: {loss: 0.2}
  - name: second
""")
    assert first.grid == second.grid == "ieee118"
    assert first.network.latency_ms == second.network.latency_ms == 25
    assert first.network.loss == 0.2
    assert second.network.loss == 0.1
