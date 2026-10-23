"""Run federate Python loops against an in-memory HELICS double (no HELICS)."""
import csv
import importlib.util
import json
import sys
import types

import numpy as np
import pytest


class HelicsDouble(types.ModuleType):
    HELICS_DATA_TYPE_STRING = 0
    HELICS_FLAG_UNINTERRUPTIBLE = 0

    def __init__(self, events):
        super().__init__('helics')
        self.events = sorted(events, key=lambda e: e[0])
        self.inputs = {}
        self.published = []
        self.t = 0

    def helicsFederateRequestTime(self, fed, target):
        self.inputs = {}
        if self.events and self.events[0][0] <= target:
            self.t, inputs = self.events.pop(0)
            self.inputs = {name: json.dumps(value) for name, value in inputs.items()}
        else:
            self.t = target
        return self.t

    def helicsInputIsUpdated(self, sub):
        return sub in self.inputs

    def helicsInputGetString(self, sub):
        return self.inputs.pop(sub, '')

    def helicsFederateRegisterGlobalPublication(self, fed, name, *args):
        return name

    def helicsFederateRegisterSubscription(self, fed, name, *args):
        return name

    def helicsPublicationPublishString(self, pub, payload):
        self.published.append((self.t, pub, json.loads(payload)))

    def __getattr__(self, name):
        if name.startswith('helics'):
            return lambda *args: None
        raise AttributeError(name)


def load_federate(name, fake, monkeypatch, repo_root):
    monkeypatch.setitem(sys.modules, 'helics', fake)
    spec = importlib.util.spec_from_file_location(name, repo_root / 'federates' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cc_config(experiment_configs, tmp_path, duration=2):
    _, config, _, _ = experiment_configs(14)
    config = dict(config, outdir=str(tmp_path), duration=duration, grid_step=0.5, bdd=True)
    config['control'] = dict(config['control'], ack_timeout_s=0.5, cooldown=0)
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(config))
    return path


def pdc_set(layout, voltage, t=0, drop_first=False):
    data = {}
    for p, ph in zip(layout.pmus, layout.measure(voltage)):
        data[str(p['bus'])] = {'ph': ph}
    if drop_first:
        del data[str(layout.pmus[0]['bus'])]
        ph = next(iter(data.values()))['ph'][0]
        ph[0] *= 1.5
        ph[1] *= 1.5
    return [{'t': t, 'release_t': t + 0.02, 'expected': len(layout.pmus), 'pmus': data}]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("missing", [True, False])
def test_cc_holds_actual_unobservable_estimate(experiment_configs, tmp_path, layout_for,
                                               monkeypatch, repo_root, missing):
    layout = layout_for(14)
    data = pdc_set(layout, np.full(14, 1.15 + 0j), drop_first=missing)
    if not missing:
        data[0]['pmus'][str(layout.pmus[0]['bus'])]['ph'][0][0] *= 1.5
    fake = HelicsDouble([(0.1, {'ns3/pdc': data})])
    cc = load_federate('cc_fed', fake, monkeypatch, repo_root)
    cc.main(str(cc_config(experiment_configs, tmp_path)))
    assert not [p for _, topic, p in fake.published if topic == 'cc/commands']
    row, = read_csv(tmp_path / 'cc_log.csv')
    assert row['observable'] == '0' and row['alarm'] == '0'
    assert row['hold_reason'] == 'unobservable'
    if not missing:
        assert row['first_alarm'] == '1' and row['removed_pmus']


def test_cc_timer_retries_during_pdc_outage(experiment_configs, tmp_path, layout_for,
                                           monkeypatch, repo_root):
    layout = layout_for(14)
    fake = HelicsDouble([(0.25, {'ns3/pdc': pdc_set(layout, np.full(14, 1.15 + 0j))})])
    cc = load_federate('cc_fed', fake, monkeypatch, repo_root)
    monkeypatch.setattr(cc, 'nearest_generators', lambda net, buses: dict.fromkeys(net.bus_ids, 2))
    cc.main(str(cc_config(experiment_configs, tmp_path)))
    attempts = [(t, p) for t, topic, p in fake.published if topic == 'cc/commands']
    assert [t for t, _ in attempts] == [0.25, 0.75]
    original, retry = [p[0] for _, p in attempts]
    assert original['id'] == retry['id'] and retry['attempt'] == 2
    rows = read_csv(tmp_path / 'cc_commands.csv')
    assert [(r['t'], r['event']) for r in rows] == [
        ('0.25', 'issued'), ('0.75', 'retried'), ('1.25', 'unacknowledged')]


@pytest.mark.parametrize(('telemetry_vset', 'expected_vset'), [
    pytest.param(None, 1.035, id='initial-value'),
    pytest.param(1.09, 1.08, id='delivered-telemetry'),
])
def test_cc_next_command_after_drop_uses_delivered_state(experiment_configs, tmp_path,
                                                        layout_for, monkeypatch, repo_root,
                                                        telemetry_vset, expected_vset):
    layout = layout_for(14)
    voltage = np.full(14, 1.15 + 0j)
    events = []
    if telemetry_vset is not None:
        events.append((0.1, {'ns3/generators': [
            dict(type='telemetry', gen_bus=2, vset=telemetry_vset, t=0),
        ]}))
    # Neither the original command nor its retry is delivered/acknowledged;
    # no generator telemetry arrives after issuance (none at all in the first case).
    events.extend([
        (0.25, {'ns3/pdc': pdc_set(layout, voltage)}),
        (1.5, {'ns3/pdc': pdc_set(layout, voltage, t=1.4)}),
    ])
    fake = HelicsDouble(events)
    cc = load_federate('cc_fed', fake, monkeypatch, repo_root)
    monkeypatch.setattr(cc, 'nearest_generators', lambda net, buses: dict.fromkeys(net.bus_ids, 2))
    cc.main(str(cc_config(experiment_configs, tmp_path, duration=1.75)))
    attempts = [(t, c) for t, topic, payload in fake.published if topic == 'cc/commands'
                for c in payload]
    assert [(t, c['id'], c['gen_bus']) for t, c in attempts] == [
        (0.25, 1, 2), (0.75, 1, 2), (1.5, 2, 2)]
    assert [r['event'] for r in read_csv(tmp_path / 'cc_commands.csv') if r['id'] == '1'] == [
        'issued', 'retried', 'unacknowledged']
    assert len(read_csv(tmp_path / 'cc_telemetry.csv')) == int(telemetry_vset is not None)
    # Gen 2 starts at 1.045 (or delivered 1.09); each decision subtracts 0.01.
    # A dropped command cannot advance that baseline: the next new id repeats
    # 1.035 (or 1.08), rather than stepping again to 1.025 (or 1.07).
    assert [c['vset'] for _, c in attempts] == [expected_vset] * 3


def test_cc_next_command_after_ack_steps_from_acknowledged_setpoint(
        experiment_configs, tmp_path, layout_for, monkeypatch, repo_root):
    layout = layout_for(14)
    voltage = np.full(14, 1.15 + 0j)
    fake = HelicsDouble([
        (0.25, {'ns3/pdc': pdc_set(layout, voltage)}),
        (0.5, {'ns3/generators': [
            dict(type='ack', id=1, gen_bus=2, vset=1.035, applied_t=0.4),
        ]}),
        (0.75, {'ns3/pdc': pdc_set(layout, voltage, t=0.7)}),
    ])
    cc = load_federate('cc_fed', fake, monkeypatch, repo_root)
    monkeypatch.setattr(cc, 'nearest_generators', lambda net, buses: dict.fromkeys(net.bus_ids, 2))
    cc.main(str(cc_config(experiment_configs, tmp_path, duration=1)))
    commands = [c for _, topic, payload in fake.published if topic == 'cc/commands'
                for c in payload]
    assert [(c['id'], c['gen_bus']) for c in commands] == [(1, 2), (2, 2)]
    assert [r['event'] for r in read_csv(tmp_path / 'cc_commands.csv') if r['id'] == '1'] == [
        'issued', 'acked']
    assert not read_csv(tmp_path / 'cc_telemetry.csv')
    # With application evidence, the next decision advances from 1.035 to 1.025.
    assert [c['vset'] for c in commands] == [1.035, 1.025]


def test_cc_uses_delivered_telemetry_and_forged_ack(experiment_configs, tmp_path, layout_for,
                                                  monkeypatch, repo_root):
    layout = layout_for(14)
    voltage = np.full(14, 1.15 + 0j)
    fake = HelicsDouble([
        (0.1, {'ns3/generators': [dict(type='telemetry', gen_bus=2, vset=1.09, q_mvar=20, t=0)]}),
        (0.25, {'ns3/pdc': pdc_set(layout, voltage)}),
        (0.5, {'ns3/generators': [dict(type='ack', id=1, gen_bus=2, vset=1.08, applied_t=0.4)]}),
        (0.6, {'ns3/generators': [dict(type='ack', id='attack:forged:2', gen_bus=2, vset=1.12, applied_t=0.55)]}),
        (0.7, {'ns3/generators': [dict(type='telemetry', gen_bus=2, vset=1.08, t=0.4)]}),
        (0.75, {'ns3/pdc': pdc_set(layout, voltage, t=0.7)}),
    ])
    cc = load_federate('cc_fed', fake, monkeypatch, repo_root)
    monkeypatch.setattr(cc, 'nearest_generators', lambda net, buses: dict.fromkeys(net.bus_ids, 2))
    cc.main(str(cc_config(experiment_configs, tmp_path, duration=1)))
    commands = [p[0] for _, topic, p in fake.published if topic == 'cc/commands']
    assert [c['vset'] for c in commands] == [1.08, 1.1]
    assert 'unexpected_ack' in [r['event'] for r in read_csv(tmp_path / 'cc_commands.csv')]
    assert [r['accepted'] for r in read_csv(tmp_path / 'cc_telemetry.csv')] == ['1', '0']


def test_grid_publishes_actual_fault_state_and_reacks_retry(experiment_configs, tmp_path,
                                                           monkeypatch, repo_root):
    _, config, _, _ = experiment_configs(14)
    config = dict(config, outdir=str(tmp_path), duration=1.5, grid_step=0.5, solver='builtin',
                  event=dict(type='avr_fault', bus=2, vset=1.12, t=0.5))
    command = dict(id=1, gen_bus=2, vset=1.07, issued_t=0.1, delivered_t=0.2, reason='test')
    fake = HelicsDouble([
        (0.5, {'ns3/cmd_delivered': [command]}),
        (1.0, {'ns3/cmd_delivered': [dict(command, attempt=2, delivered_t=0.9)]}),
        (1.5, {'ns3/cmd_delivered': [dict(command, id='attack:forged:2', vset=1.11,
                                      reason='attack: forged command')]}),
    ])
    grid = load_federate('grid_fed', fake, monkeypatch, repo_root)
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(config))
    grid.main(str(path))
    reports = [p for _, topic, p in fake.published if topic == 'grid/generators']
    assert len(reports) == 4
    at_fault = reports[1]
    state = next(g for g in at_fault['generators'] if g['gen_bus'] == 2)
    assert state['vset'] == 1.12 and np.isfinite(state['q_mvar'])
    assert at_fault['acks'] == [dict(id=1, gen_bus=2, vset=1.07, applied_t=0.5)]
    assert reports[2]['acks'] == at_fault['acks']
    assert next(g for g in reports[2]['generators'] if g['gen_bus'] == 2)['vset'] == 1.12
    assert reports[3]['acks'][0]['id'] == 'attack:forged:2'
    applied = read_csv(tmp_path / 'commands_applied.csv')
    assert [r['id'] for r in applied] == ['1', '', 'attack:forged:2']
