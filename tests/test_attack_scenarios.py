"""WP11 schema, CLI and pure attack config regression coverage (no HELICS)."""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from scenario import Attack, Scenario, parse
import run_experiment as run
from cpslib.attack import plan_fdi
from cpslib.control import nearest_generators

TYPES = ['none', 'fdi-simple', 'fdi-stealthy', 'drop', 'delay', 'replay',
         'cmd-drop', 'cmd-delay', 'cmd-inject', 'coordinated']


@pytest.mark.parametrize('kind', TYPES)
def test_schema_and_cli_attack_fields(kind):
    attack = Attack(type=kind, start=3, end=8, record_s=2, ramp_s=1, vset=1.12)
    kwargs = Scenario(attack=attack).run_args()
    assert kwargs['record_s'] == 2
    assert kwargs['ramp_s'] == 1
    assert kwargs['attack_vset'] == 1.12
    args = run.parse_args(['--attack', kind, '--record-s', '2', '--ramp-s', '1', '--attack-vset', '1.12'])
    assert (args.attack, args.record_s, args.ramp_s, args.attack_vset) == (kind, 2, 1, 1.12)


@pytest.mark.parametrize('fields', [
    {'type': 'cmd-inject', 'vset': 1.5}, {'type': 'coordinated', 'vset': 0.79},
    {'type': 'replay', 'record_s': 5}, {'type': 'replay', 'record_s': 0},
    {'type': 'replay', 'start': 0}, {'type': 'fdi-simple', 'ramp_s': -1},
    *[{'type': kind, 'start': 4, 'end': end} for kind in TYPES[1:] for end in [3, 4]],
])
def test_invalid_attack_fields(fields):
    with pytest.raises(ValidationError):
        Attack(**fields)


@pytest.mark.parametrize('case', [14, 118])
def test_new_scenario_files(case, repo_root, network_for, layout_for):
    scenarios = parse((repo_root / f'scenarios/attacks-ieee{case}.yaml').read_text())
    assert {s.attack.type for s in scenarios} == set(TYPES)
    for kind in ['fdi-simple', 'replay']:
        assert {s.pmu.placement for s in scenarios if s.attack.type == kind} == {'minimum', 'redundant'}
    for s in scenarios:
        assert s.solver == 'builtin' and s.duration == 10
        if s.attack.type in ['cmd-drop', 'cmd-delay']:
            assert s.event.type == 'avr_fault' and s.event.bus == s.attack.target
        args = run.parse_args([])
        for key, value in s.run_args().items():
            if key != 'case':
                setattr(args, key, value)
        net = network_for(case)
        layout = layout_for(case, 1 if s.pmu.placement == 'minimum' else 2)
        voltage, _, ok = net.solve_pf()
        assert ok
        run.build_attack(args, layout, voltage, run.parse_event(args.event))


def build(kind, layout_for, solution_for, *extra):
    args = run.parse_args(['--attack', kind, '--target', '8', '--fake', '1.05', *extra])
    return run.build_attack(args, layout_for(14), solution_for(14), run.parse_event(args.event))


@pytest.mark.parametrize('kind', TYPES[:5])
@pytest.mark.parametrize('event', ['none', 'avr:6:1.12:3', 'load:4:20:3'])
def test_legacy_dicts_are_identical(kind, event, tmp_path, repo_root):
    # Captured by calling base-branch build_configs, before production edits.
    frozen = json.loads((repo_root / 'tests/fixtures/wp11_legacy_attacks.json').read_text())
    args = run.parse_args(['--attack', kind, '--target', '8', '--fake', '1.05', '--event', event])
    cluster = {'placement': {}, 'head': 'localhost', 'nodes': [], 'mpi': {}}
    meta = run.build_configs(args, str(tmp_path), cluster, 23400)
    attack = json.loads((tmp_path / 'ns3_config.json').read_text())['attack']
    assert attack == frozen[kind + '|' + event]
    assert meta['target_bus'] == 8


@pytest.mark.parametrize('kind', ['fdi-simple', 'fdi-stealthy'])
def test_fdi_ramp_opt_in(kind, layout_for, solution_for):
    step, _, _ = build(kind, layout_for, solution_for)
    ramp, _, _ = build(kind, layout_for, solution_for, '--ramp-s', '2')
    assert ramp == dict(step, ramp_s=2)


@pytest.mark.parametrize('target,targets', [(None, [2, 7, 11, 13]), (8, [7])])
def test_replay_scope_and_defaults(target, targets, layout_for, solution_for):
    args = run.parse_args(['--attack', 'replay'])
    args.target = target
    attack, plan, _ = run.build_attack(args, layout_for(14), solution_for(14), {'type': 'none'})
    assert attack == {'type': 'replay', 'targets': targets, 'record_s': 3, 'start': 3, 'end': 8}
    assert plan is None
    args.attack_end = 4
    attack, _, _ = run.build_attack(args, layout_for(14), solution_for(14), {'type': 'none'})
    assert attack['record_s'] == 1
    args.record_s = 0.5
    attack, _, _ = run.build_attack(args, layout_for(14), solution_for(14), {'type': 'none'})
    assert attack['record_s'] == 0.5


@pytest.mark.parametrize('kind', ['cmd-drop', 'cmd-delay', 'cmd-inject', 'coordinated'])
def test_command_target_rules(kind, layout_for, solution_for):
    args = run.parse_args(['--attack', kind])
    layout, voltage = layout_for(14), solution_for(14)
    attack, _, target = run.build_attack(args, layout, voltage, {'type': 'none'})
    default_fdi_target = layout.net.bus_ids[int(np.argmax(np.abs(voltage)))]
    generators = {g['bus'] for g in layout.net.gens if g['status']}
    assert target == nearest_generators(layout.net, generators)[default_fdi_target]
    assert attack['gen_targets'] == [target]
    for invalid in [0, 4, 99999]:
        args.target = invalid
        with pytest.raises(ValueError, match='generator bus'):
            run.build_attack(args, layout, voltage, {'type': 'none'})


@pytest.mark.parametrize('kind', ['cmd-drop', 'cmd-delay', 'cmd-inject'])
def test_command_dict(kind, layout_for, solution_for):
    attack, plan, _ = build(kind, layout_for, solution_for, '--attack-vset', '1.12')
    expected = {'type': kind, 'targets': [], 'gen_targets': [8], 'start': 3, 'end': 8}
    expected.update({'vset': 1.12} if kind == 'cmd-inject' else {'delay_ms': 100})
    assert attack == expected and plan is None


def test_coordinated_plans_after_forgery(layout_for, solution_for, monkeypatch):
    layout = layout_for(14, 2)  # includes the generator PMU: adaptive ref must survive
    args = run.parse_args(['--attack', 'coordinated', '--target', '8', '--attack-vset', '1.12'])
    pre = solution_for(14)
    after, _, ok = layout.net.solve_pf(vset={8: 1.12})
    assert ok
    expected = plan_fdi(layout, after, 8, abs(pre[layout.net.idx[8]]), 'stealthy')
    actual_planner = run.plan_fdi
    calls = []
    def spy(layout, voltage, target, fake, mode):
        calls.append((voltage.copy(), mode))
        return actual_planner(layout, voltage, target, fake, mode)
    monkeypatch.setattr(run, 'plan_fdi', spy)
    attack, plan, target = run.build_attack(args, layout, pre, {'type': 'none'})
    np.testing.assert_allclose(calls[0][0], after, rtol=0, atol=0)
    assert calls[0][1] == 'stealthy'
    assert plan == expected
    assert attack == {'type': 'coordinated', 'targets': plan['compromised_pmus'],
                      'gen_targets': [8], 'vset': 1.12, 'deltas': plan['deltas'],
                      'ref': {'bus': 8, 'ch': 0}, 'v_fake': abs(pre[layout.net.idx[8]]),
                      'start': 3, 'end': 8}
    assert target == 8


@pytest.mark.parametrize('opts,error', [
    (['--record-s', '4', '--attack', 'replay'], 'record_s'),
    (['--attack', 'replay', '--attack-start', '0'], 'record_s'),
    (['--ramp-s', '-1'], 'ramp_s'), (['--attack-vset', '1.5'], 'vset'),
    (['--attack', 'cmd-drop', '--attack-end', '3'], 'end'),
])
def test_cli_cross_field_validation(opts, error, layout_for, solution_for):
    args = run.parse_args(opts)
    with pytest.raises(ValueError, match=error):
        run.build_attack(args, layout_for(14), solution_for(14), {'type': 'none'})


@pytest.mark.parametrize('kind', TYPES)
def test_build_configs_wires_attack_helper(kind, tmp_path, layout_for, solution_for):
    args = run.parse_args(['--attack', kind, '--target', '8', '--fake', '1.05'])
    expected, plan, target = run.build_attack(args, layout_for(14), solution_for(14), {'type': 'none'})
    cluster = {'placement': {}, 'head': 'localhost', 'nodes': [], 'mpi': {}}
    meta = run.build_configs(args, str(tmp_path), cluster, 23400)
    actual = json.loads((tmp_path / 'ns3_config.json').read_text())['attack']
    assert actual == json.loads(json.dumps(expected))
    assert meta['attack_plan'] == plan and meta['target_bus'] == target


@pytest.mark.parametrize('event', [
    {'type': 'avr_fault', 'bus': 6, 'vset': 1.1, 't': 1},
    {'type': 'avr_fault', 'bus': 8, 'vset': 1.1, 't': 1},
    {'type': 'avr_fault', 'bus': 8, 'vset': 1.1, 't': 4},
    {'type': 'load_step', 'bus': 4, 'pct': 20, 't': 1},
    {'type': 'load_step', 'bus': 4, 'pct': 20, 't': 4},
])
def test_coordinated_event_operating_point(event, layout_for, solution_for):
    layout = layout_for(14)
    net = layout.net
    args = run.parse_args(['--attack', 'coordinated', '--target', '8', '--attack-vset', '1.12'])
    vset, pd, qd = {}, None, None
    if event['t'] < args.attack_start:
        if event['type'] == 'avr_fault':
            vset[event['bus']] = event['vset']
        else:
            pd = np.array([b['pd'] for b in net.buses])
            qd = np.array([b['qd'] for b in net.buses])
            pd[net.idx[4]] *= 1.2; qd[net.idx[4]] *= 1.2
    pre, _, ok_pre = net.solve_pf(pd, qd, vset=vset)
    vset[8] = 1.12
    after, _, ok_after = net.solve_pf(pd, qd, vset=vset)
    assert ok_pre and ok_after
    _, plan, _ = run.build_attack(args, layout, solution_for(14), event)
    assert plan == plan_fdi(layout, after, 8, abs(pre[net.idx[8]]), mode='stealthy')


@pytest.fixture(scope='module')
def request_model(repo_root):
    # Load only the real API model: app import starts job threads and needs FastAPI.
    import ast
    from types import SimpleNamespace
    from typing import Optional
    from pydantic import BaseModel, Field, model_validator
    import scenario
    tree = ast.parse((repo_root / 'webapp/app.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RunRequest')
    namespace = {'BaseModel': BaseModel, 'Field': Field, 'model_validator': model_validator,
                 'Optional': Optional, 'scen': scenario, 'grids': SimpleNamespace(BUILTIN={'14'})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'webapp/app.py', 'exec'), namespace)
    return namespace['RunRequest']


@pytest.mark.parametrize('kind', TYPES)
def test_api_to_cli_roundtrip(kind, request_model):
    request = request_model(attack=kind, record_s=1, ramp_s=2, attack_vset=1.11)
    args = run.parse_args(request.argv())
    assert (args.attack, args.record_s, args.ramp_s, args.attack_vset) == (kind, 1, 2, 1.11)


@pytest.mark.parametrize('start,end,expected', [(3, 8, 3), (4, 5, 1)])
def test_replay_default_resolves_in_schema_and_api(start, end, expected, request_model):
    attack = Attack(type='replay', start=start, end=end)
    assert attack.record_s == expected
    assert Scenario(attack=attack).run_args()['record_s'] == expected
    request = request_model(attack='replay', attack_start=start, attack_end=end)
    assert request.record_s == expected
    assert run.parse_args(request.argv()).record_s == expected


@pytest.mark.parametrize('fields', [
    {'attack': 'replay', 'record_s': 4}, {'attack': 'replay', 'attack_start': 0},
    {'attack': 'cmd-inject', 'attack_vset': 1.5}, {'ramp_s': -1},
    {'attack': 'coordinated', 'attack_end': 3},
])
def test_api_validates_attack_fields(fields, request_model):
    with pytest.raises(ValidationError):
        request_model(**fields)


@pytest.mark.parametrize('kind', TYPES)
def test_summary_appends_attack_counters(kind, tmp_path):
    import csv
    args = run.parse_args(['--attack', kind, '--target', '8'])
    args.grid_step = 0.5
    # Isolate reporting from planning, so missing fields fail directly on the old analyzer.
    attack = {'type': kind, 'targets': [7] if kind.startswith('fdi') or kind in ['drop', 'delay', 'replay', 'coordinated'] else [],
              'gen_targets': [8], 'vset': 1.12, 'record_s': 3}
    (tmp_path / 'ns3_config.json').write_text(json.dumps({'attack': attack}))
    counters = {'cmds_forged': 1, 'cmds_dropped': 2, 'cmds_delayed': 3, 'replayed': 4}
    (tmp_path / 'ns3_summary.json').write_text(json.dumps(counters))
    rows = {
        'grid_truth.csv': [{'t': 0, 'v8': 1.09}, {'t': 4, 'v8': 1.12}],
        'grid_steps.csv': [{'solver': 'builtin', 'converged': 'True'}],
        'cc_log.csv': [{'t': 4, 'release_t': 4.02, 'violations': 0, 'first_alarm': 0,
                        'pmus_received': 4, 'pmus_expected': 4, 'observable': 1}],
        'cc_estimates.csv': [{'t': 0, 'v8': 1.09}, {'t': 4, 'v8': 1.09}],
        'commands_applied.csv': [{'t_applied': 4, 'gen_bus': 8, 'vset': 1.12, 'issued_t': 3,
                                  'delivered_t': 3.01, 'reason': 'attack: forged command'}],
    }
    for name, data in rows.items():
        with (tmp_path / name).open('w') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    meta = {'args': vars(args), 'grid_name': 'IEEE 14-bus', 'cluster': {},
            'target_bus': 8, 'attack_plan': None}
    summary = run.analyze(str(tmp_path), meta, {}, 1.0)
    result = summary['attack']
    assert result['type'] == kind
    for field, expected in [('forged_commands', 1), ('commands_dropped', 2),
                            ('commands_delayed', 3), ('replayed_frames', 4)]:
        assert result[field] == expected
    if kind == 'replay':
        assert result['replay_scope'] == 'target_bus_observers' and result['record_s'] == 3
        meta['args']['target'] = None
        assert run.analyze(str(tmp_path), meta, {}, 1)['attack']['replay_scope'] == 'all_pmus'
    if kind.startswith('cmd-') or kind == 'coordinated':
        assert result['generator_targets'] == [8] and result['compromised_pmus'] == attack['targets']
    if kind in ['cmd-inject', 'coordinated']:
        assert result['vset'] == 1.12 and 'CC socket' in result['forged_source']
    # Older NS-3 summaries still work and default the appended counters to zero.
    (tmp_path / 'ns3_summary.json').write_text('{}')
    result = run.analyze(str(tmp_path), meta, {}, 1)['attack']
    assert all(result[k] == 0 for k in ['forged_commands', 'commands_dropped', 'commands_delayed', 'replayed_frames'])


def test_dashboard_attack_controls(repo_root):
    from html.parser import HTMLParser
    class Controls(HTMLParser):
        def __init__(self):
            super().__init__()
            self.options, self.fields = set(), set()
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'option':
                self.options.add(attrs.get('value'))
            if tag == 'input':
                self.fields.add(attrs.get('name') or attrs.get('v-model.number'))
    web = Controls()
    web.feed((repo_root / 'webapp/index.html').read_text())
    assert set(TYPES) <= web.options
    assert {'record_s', 'ramp_s', 'attack_vset'} <= web.fields
    nodes = json.loads((repo_root / 'node-red/flows.json').read_text())
    node = next(n for n in nodes if 'Cyber attack' in n.get('format', ''))
    dashboard = Controls()
    dashboard.feed(node['format'])
    assert set(TYPES) <= dashboard.options
    assert {'f.attack.record_s', 'f.attack.ramp_s', 'f.attack.vset'} <= dashboard.fields
