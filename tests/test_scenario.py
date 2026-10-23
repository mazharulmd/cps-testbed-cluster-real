import pytest

from scenario import Scenario, parse


@pytest.mark.parametrize(
    "path,count", [("webapp/scenario_template.yaml", 1), ("scenarios/demo-ieee14.yaml", 3)]
)
def test_shipped_scenarios_parse(path, count, repo_root):
    scenarios = parse((repo_root / path).read_text())
    assert len(scenarios) == count
    assert all(isinstance(item, Scenario) for item in scenarios)


def test_all_scenario_files_parse(repo_root):
    for path in (repo_root / "scenarios").glob("*.yaml"):
        assert parse(path.read_text()), path


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
        ("pmu: {noise_mag: -0.001}", "pmu.noise_mag"),
        ("pmu: {noise_ang: 0.051}", "pmu.noise_ang"),
        ("pmu: {noise_mag: .nan}", "pmu.noise_mag"),
        ("control: {se_sigma: 0}", "control.se_sigma"),
        ("control: {se_sigma: -0.001}", "control.se_sigma"),
        ("control: {se_sigma: .inf}", "control.se_sigma"),
        ("control: {se_sigma: .nan}", "control.se_sigma"),
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


@pytest.fixture
def run_request_class(run_request_model):
    return run_request_model


@pytest.mark.parametrize(
    "text,mag,ang,sigma",
    [("{}", 0.001, 0.001, None),
     ("pmu: {noise_mag: 0.003, noise_ang: 0.004}", 0.003, 0.004, None),
     ("pmu: {noise_mag: 0, noise_ang: 0}\ncontrol: {se_sigma: 0.007}", 0, 0, 0.007)],
)
def test_scenario_noise_reaches_cli(text, mag, ang, sigma, run_request_class):
    from run_experiment import parse_args

    scenario = parse(text)[0]
    fields = scenario.run_args()
    assert fields["noise_mag"] == mag
    assert fields["noise_ang"] == ang
    assert fields["se_sigma"] == sigma
    request = run_request_class(**fields)
    assert request.model_dump()["noise_mag"] == mag
    assert request.model_dump()["noise_ang"] == ang
    assert request.model_dump()["se_sigma"] == sigma
    argv = request.argv()
    args = parse_args(argv)
    assert args.noise_mag == mag
    assert args.noise_ang == ang
    assert args.se_sigma == sigma
    assert ("--se-sigma" in argv) == (sigma is not None)


def test_scenario_noise_nested_defaults():
    first, second = parse("""
pmu: {noise_mag: 0.003, noise_ang: 0.004}
control: {se_sigma: 0.007}
scenarios:
  - name: derived
    pmu: {noise_ang: 0.002}
    control: {se_sigma: null}
  - name: override
""")
    assert first.pmu.noise_mag == second.pmu.noise_mag == 0.003
    assert first.pmu.noise_ang == 0.002
    assert first.control.se_sigma is None
    assert second.pmu.noise_ang == 0.004
    assert second.control.se_sigma == 0.007
