"""WP4 control safety and protocol regressions; no HELICS runtime required."""
import pytest


def test_unobservable_gross_error_is_held(layout_for, solution_for):
    from cpslib.control import control_decision
    from cpslib.estimation import StateEstimator

    layout = layout_for(14)
    z = layout.H @ solution_for(14)
    meas = {key: z[row] for key, row in layout.row.items() if key[0] != 0}
    meas[next(iter(meas))] *= 1.5
    result = StateEstimator(layout, sigma=0.001).estimate_with_bdd(meas)
    assert not result['observable'] and not result['alarm']  # preserve metric semantics
    assert result['pseudo_measurements'] == layout.net.n
    assert control_decision(result) == 'unobservable'


@pytest.mark.parametrize('result,options,reason', [
    ({'observable': False, 'alarm': False}, {}, 'unobservable'),
    ({'observable': True, 'pseudo_measurements': 1, 'alarm': False}, {}, 'pseudo_measurements'),
    ({'observable': True, 'alarm': True, 'removed_pmus': [0]}, {}, 'bad_data'),
    ({'observable': False, 'alarm': False, 'removed_pmus': [0]}, {}, 'unobservable'),
    ({'observable': True, 'alarm': False, 'first_alarm': True, 'removed_pmus': [0]}, {}, ''),
    ({'observable': True, 'alarm': False}, {}, ''),
    ({'observable': True, 'alarm': False}, {'enabled': False}, 'disabled'),
    ({'observable': True, 'alarm': False}, {'cooldown': True}, 'cooldown'),
])
def test_control_decision(result, options, reason):
    from cpslib.control import control_decision
    assert control_decision(result, **options) == reason


def test_commands_retry_once_then_unacked():
    from cpslib.control import CommandTracker
    tracker = CommandTracker(2)
    command = {'id': 1, 'gen_bus': 2, 'vset': 1.04, 'issued_t': 0}
    tracker.issue(command, 0)
    assert tracker.pending_for(2)
    assert tracker.next_deadline(100) == 2
    assert tracker.expire(1.999) == []
    retry, = tracker.expire(2)
    assert retry['id'] == 1 and retry['vset'] == 1.04
    assert retry['attempt'] == 2  # distinct payload survives NS-3 publication deduplication
    assert command == {'id': 1, 'gen_bus': 2, 'vset': 1.04, 'issued_t': 0}
    assert tracker.expire(3.999) == []
    assert tracker.expire(4) == []
    assert not tracker.pending_for(2)
    assert tracker.expire(100) == []
    assert tracker.counts == dict(commands_acked=0, commands_retried=1,
                                 commands_unacked=1, unexpected_acks=0)
    assert [e['event'] for e in tracker.events] == ['issued', 'retried', 'unacknowledged']


def test_ack_and_unknown_forgery():
    from cpslib.control import CommandTracker
    tracker = CommandTracker(2)
    tracker.issue({'id': 7, 'gen_bus': 8, 'vset': 1.03}, 0)
    assert tracker.acknowledge({'id': 'attack:forged:8', 'gen_bus': 8}, 0.5) == 'unexpected_ack'
    assert tracker.pending_for(8)
    assert tracker.acknowledge({'id': 7, 'gen_bus': 2}, 1) == 'unexpected_ack'
    assert tracker.acknowledge({'id': 7, 'gen_bus': 8}, 2) == 'acked'
    assert tracker.expire(2) == []  # process ack before the timeout at the same grant
    assert tracker.acknowledge({'id': 7, 'gen_bus': 8}, 3) == 'duplicate_ack'
    assert tracker.counts == dict(commands_acked=1, commands_retried=0,
                                 commands_unacked=0, unexpected_acks=2)


def test_late_ack_and_end_of_run_accounting():
    from cpslib.control import CommandTracker
    tracker = CommandTracker(1)
    tracker.issue({'id': 1, 'gen_bus': 2}, 0)
    tracker.expire(1)
    tracker.expire(2)
    assert tracker.acknowledge({'id': 1, 'gen_bus': 2}, 3) == 'late_ack'
    tracker.issue({'id': 2, 'gen_bus': 2}, 3)
    tracker.finish(3.5)
    tracker.finish(3.5)
    assert tracker.counts == dict(commands_acked=1, commands_retried=1,
                                 commands_unacked=1, unexpected_acks=0)
    assert tracker.events[-1]['reason'] == 'run_end'


def test_telemetry_reconciles_faults_forgery_and_dropped_commands():
    from cpslib.control import CommandTracker, SetpointTelemetry
    state = SetpointTelemetry({2: 1.0, 8: 1.02})
    tracker = CommandTracker(2)
    tracker.issue({'id': 1, 'gen_bus': 2, 'vset': 1.01}, 0)
    tracker.expire(2)
    assert state.vset[2] == 1.0  # a lost command changes no known applied setpoint
    assert state.update({'gen_bus': 2, 'vset': 1.12, 'q_mvar': 25, 't': 2})
    assert state.vset[2] == 1.12 and state.q_mvar[2] == 25
    assert not state.update({'gen_bus': 2, 'vset': 1.0, 't': 1})
    assert state.vset[2] == 1.12
    assert state.update({'gen_bus': 8, 'vset': 1.15, 't': 3})
    assert state.vset[8] == 1.15
    assert state.update({'gen_bus': 8, 'vset': 1.15, 't': 4, 'q_pu_by_machine': {'1': 0.25, '2': 0.1}})
    assert state.q_pu_by_machine[8] == {'1': 0.25, '2': 0.1}
    assert not state.update({'gen_bus': 999, 'vset': 1.1, 't': 3})


def test_grid_reacknowledges_retry_without_reapplying():
    from cpslib.control import AppliedCommands
    grid = AppliedCommands()
    cmd = {'id': 1, 'gen_bus': 2, 'vset': 1.01}
    apply, ack = grid.receive(cmd, 0.5)
    assert apply and ack == dict(id=1, gen_bus=2, vset=1.01, applied_t=0.5)
    apply, duplicate_ack = grid.receive(dict(cmd, attempt=2), 3)
    assert not apply and duplicate_ack == ack
    forged = dict(cmd, id='attack:forged:2', vset=1.12)
    apply, ack = grid.receive(forged, 4)
    assert apply and ack['id'] == 'attack:forged:2' and ack['vset'] == 1.12


@pytest.mark.parametrize('options', [(), ('--telemetry-s', '0.75', '--ack-timeout-s', '3.25')])
def test_protocol_config_defaults_and_overrides(options, experiment_configs):
    from cpslib.control import default_ack_timeout
    ns3, run, meta, _ = experiment_configs(14, options)
    ctl = run['control']
    assert ctl['telemetry_s'] == ns3['telemetry_s'] == (0.75 if options else 2)
    expected = 3.25 if options else default_ack_timeout(
        max(g['delay_ms'] for g in ns3['gens']), 0.5, ns3['rate'])
    assert ctl['ack_timeout_s'] == pytest.approx(expected)
    assert meta['control_center']['ack_timeout_s'] == ctl['ack_timeout_s']


def test_protocol_scenario_api_cli(run_request_model):
    from scenario import parse
    from run_experiment import parse_args
    args = parse_args(run_request_model(**parse(
        'control: {telemetry_s: 0.75, ack_timeout_s: 3.25}')[0].run_args()).argv())
    assert args.telemetry_s == 0.75 and args.ack_timeout_s == 3.25
    assert parse('{}')[0].control.telemetry_s == 2
    assert parse('{}')[0].control.ack_timeout_s is None


@pytest.mark.parametrize('name', ['telemetry_s', 'ack_timeout_s'])
@pytest.mark.parametrize('value', ['0', '-1', '.nan', '.inf'])
def test_protocol_rejects_invalid_intervals(name, value):
    from scenario import parse
    from run_experiment import parse_args
    with pytest.raises(ValueError):
        parse(f'control: {{{name}: {value}}}')
    with pytest.raises(SystemExit):
        parse_args(['--' + name.replace('_', '-'), value.lstrip('.') if value.startswith('.') else value])


def test_command_summary_and_hold_reasons(experiment_configs, monkeypatch):
    from pathlib import Path
    import run_experiment as run
    _, _, meta, out = experiment_configs(14)
    meta['args']['grid_step'] = 0.5
    (out / 'ns3_summary.json').write_text('{}')
    (out / 'cc_commands.csv').write_text(
        't,event,id,gen_bus,attempt,reason\n0,issued,1,2,1,\n1,retried,1,2,2,\n'
        '2,unacknowledged,1,2,2,timeout\n3,unexpected_ack,attack:forged:2,2,,\n'
        '4,issued,2,8,1,\n5,acked,2,8,1,\n')
    rows = {'grid_truth.csv': [{'t': '0', 'v1': '1'}],
            'cc_log.csv': [dict(t='0', release_t='0', violations='0', first_alarm='0',
                                pmus_received='0', pmus_expected='1', observable='0', hold_reason='bad_data')]}
    actual_read = run.read_csv
    monkeypatch.setattr(run, 'read_csv', lambda p: actual_read(p) if Path(p).name == 'cc_commands.csv'
                        else rows.get(Path(p).name, []))
    summary = run.analyze(str(out), meta, {}, 0)
    assert summary['commands'] == dict(commands_acked=1, commands_retried=1,
                                       commands_unacked=1, unexpected_acks=1)
    assert summary['control_center']['hold_reason_counts'] == {'bad_data': 1}
    (out / 'cc_commands.csv').unlink()
    assert run.analyze(str(out), meta, {}, 0)['commands'] == dict(
        commands_acked=0, commands_retried=0, commands_unacked=0, unexpected_acks=0)


def test_bdd_removal_that_loses_observability_is_held(layout_for, solution_for):
    from cpslib.control import control_decision
    from cpslib.estimation import StateEstimator
    layout = layout_for(14)
    z = layout.H @ solution_for(14)
    measurements = {key: z[row] for key, row in layout.row.items()}
    measurements[(0, 0)] *= 1.5
    result = StateEstimator(layout, sigma=0.001).estimate_with_bdd(measurements)
    assert result['first_alarm'] and result['removed_pmus']
    assert not result['observable'] and not result['alarm']
    assert result['pseudo_measurements'] == 14
    assert control_decision(result) == 'unobservable'
