"""Reactive limits, independent power-flow checks, and federate option wiring."""
import ast
import csv
import importlib.util
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

from pydantic import BaseModel, Field

import numpy as np
import pytest

from cpslib.network import Network
from cpslib.gridpack import GridPackMPISolver
from run_experiment import parse_args, build_configs
from scenario import parse


def qg(net, voltage, qd=None):
    if qd is None:
        qd = [b["qd"] for b in net.buses]
    return (voltage * np.conj(net.Ybus @ voltage)).imag * net.base + qd


def test_avr_limit_and_reset(network_for):
    net = network_for(118)
    voltage, iterations, ok = net.solve_pf(vset={49: 1.12})
    assert ok and iterations > 0
    assert qg(net, voltage)[net.idx[49]] == pytest.approx(210, abs=1e-3)
    assert abs(voltage[net.idx[49]]) < 1.12
    assert net.last_qlim[49] == "max"
    assert net.q_limits_at(voltage)[49] == "max"
    assert all(net.buses[net.idx[b]]["type"] == 2 for b in net.last_qlim)
    # Each call starts with the original PV buses, including after a warm start.
    base, _, ok = net.solve_pf(V_init=voltage)
    assert ok and 49 not in net.last_qlim
    plain, _, ok = net.solve_pf(vset={49: 1.12}, qlim=False)
    assert ok and net.last_qlim == {}
    assert abs(plain[net.idx[49]]) == pytest.approx(1.12, abs=1e-14)
    assert qg(net, plain)[net.idx[49]] > 700
    assert not np.allclose(base, plain)


def test_iteration_failures(network_for):
    net = network_for(118)
    _, _, ok = net.solve_pf(vset={49: 1.12}, max_qlim_it=1)
    assert not ok  # An unconstrained convergence alone is insufficient.
    _, it, ok = net.solve_pf(max_it=0)
    assert not ok and it == 0 and net.last_qlim == {}


def test_aggregate_online_limits_and_actual_load(network_for):
    # Split a machine into two online units and an offline unit with huge limits.
    source = network_for(118)
    topo = deepcopy(dict(baseMVA=source.base, buses=source.buses,
                         gens=source.gens, branches=source.branches))
    original = next(g for g in topo["gens"] if g["bus"] == 49)
    original.update(pg=original["pg"] / 2, qmin=original["qmin"] / 2,
                    qmax=original["qmax"] / 2)
    topo["gens"] += [dict(original), dict(original, status=0, qmin=-99999, qmax=99999)]
    net = Network(topo)
    pd = np.array([b["pd"] for b in net.buses]) * 1.01
    qd = np.array([b["qd"] for b in net.buses]) * 1.15
    voltage, _, ok = net.solve_pf(pd, qd, {49: 1.12})
    assert ok and net.last_qlim[49] == "max"
    assert qg(net, voltage, qd)[net.idx[49]] == pytest.approx(210, abs=1e-3)
    assert net.q_limits_at(voltage, qd)[49] == "max"


def test_lower_limit_and_slack_exclusion(network_for):
    net = network_for(39)
    voltage, _, ok = net.solve_pf()
    assert ok and net.last_qlim[37] == "min"
    assert qg(net, voltage)[net.idx[37]] == pytest.approx(0, abs=1e-3)
    slack = next(b["id"] for b in net.buses if b["type"] == 3)
    assert slack not in net.last_qlim and slack not in net.q_limits_at(voltage)


def pandapower_net(net):
    """Convert our actual topology, including shunts, taps, and online status."""
    pytest.importorskip("pandapower")
    from pandapower.converter.pypower import from_ppc
    bus = np.zeros((net.n, 13))
    for i, b in enumerate(net.buses):
        bus[i] = [b["id"], b["type"], b["pd"], b["qd"], b["gs"], b["bs"],
                  b.get("area", 1), b.get("vm", 1), b.get("va", 0),
                  b.get("base_kv", 0) or 100, 1, 1.5, 0.5]
    gen = np.zeros((len(net.gens), 21))
    for i, g in enumerate(net.gens):
        gen[i, :10] = [g["bus"], g["pg"], 0, g["qmax"], g["qmin"], g["vg"],
                       net.base, g["status"], 1e9, -1e9]
    branch = np.zeros((len(net.branches), 13))
    for i, b in enumerate(net.branches):
        branch[i] = [b["from"], b["to"], b["r"], b["x"], b["b"], 0, 0, 0,
                     b["tap"], b["shift"], b["status"], -360, 360]
    return from_ppc(dict(baseMVA=net.base, bus=bus, gen=gen, branch=branch), f_hz=60)


@pytest.mark.parametrize("case,vset", [(14, {}), (118, {}), (118, {49: 1.12})])
def test_pandapower_agreement(case, vset, network_for):
    pp = pytest.importorskip("pandapower")
    net = network_for(case)
    other = pandapower_net(net)
    # from_ppc uses original bus IDs as the pandapower bus indices.
    for bus, value in vset.items():
        other.gen.loc[other.gen.bus == bus, "vm_pu"] = value
    for enforce in (False, True):
        pp.runpp(other, enforce_q_lims=enforce, numba=False, trafo_model="pi",
                 calculate_voltage_angles=True, init="flat", tolerance_mva=1e-9)
        voltage, _, ok = net.solve_pf(vset=vset, qlim=enforce)
        error = np.max(np.abs(np.abs(voltage) - other.res_bus.vm_pu.to_numpy()))
        print(f"IEEE {case} {vset=} {enforce=}: max |dV|={error:.3e} pu")
        assert ok and other.converged
        assert error <= 1e-4


@pytest.mark.parametrize("enforce", [True, False])
def test_options_and_config(enforce, tmp_path, repo_root):
    scenario = parse(f"grid: ieee118\nqlim: {str(enforce).lower()}")[0]
    assert scenario.run_args()["qlim"] is enforce
    args = parse_args(["--case", "118"] + ([] if enforce else ["--no-qlim"]))
    args.mpi_np, args.grid_step = 1, 0.5
    assert args.qlim is enforce
    cluster = dict(head="localhost", nodes=[], placement={}, mpi={})
    build_configs(args, str(tmp_path), cluster, 23000)
    cfg = json.loads((tmp_path / "run_config.json").read_text())
    assert cfg["qlim"] is enforce
    assert parse("grid: ieee118")[0].qlim is True


@pytest.mark.parametrize("enforce", [True, False])
def test_gridpack_xml_and_limit_diagnostics(enforce, monkeypatch, tmp_path, network_for):
    net = network_for(118)
    topo = dict(baseMVA=net.base, buses=net.buses, gens=net.gens, branches=net.branches)
    monkeypatch.setattr(GridPackMPISolver, "_start", lambda *args: None)
    solver = GridPackMPISolver(net, topo, str(tmp_path), qlim=enforce)
    assert f"<qlim>{int(enforce)}</qlim>" in (tmp_path / "input.xml").read_text()
    voltage, _, _ = net.solve_pf(vset={49: 1.12}, qlim=enforce)
    def read(tag, collect=None):
        if collect is not None:
            collect["@@V"] = [[str(b), str(abs(v)), str(np.rad2deg(np.angle(v)))]
                              for b, v in zip(net.bus_ids, voltage)]
        return {"ok": "1"}
    monkeypatch.setattr(solver, "_send", lambda lines: None)
    monkeypatch.setattr(solver, "_read_until", read)
    _, _, ok = solver.solve([b["pd"] for b in net.buses], [b["qd"] for b in net.buses], {49: 1.12})
    assert ok
    assert solver.last_qlim.get(49) == ("max" if enforce else None)


@pytest.mark.parametrize("enforce", [None, True, False])
@pytest.mark.parametrize("backend", ["builtin", "gridpack", "gridpack-fails"])
def test_grid_federate_avr_csv(enforce, backend, tmp_path, repo_root, monkeypatch):
    # Exercise the real step loop without HELICS/NS-3. No commands are delivered.
    fake_h = SimpleNamespace(HELICS_FLAG_UNINTERRUPTIBLE=1, HELICS_DATA_TYPE_STRING=1,
                             helicsInputIsUpdated=lambda *args: False)
    names = ("CreateFederateInfo FederateInfoSetCoreTypeFromString FederateInfoSetCoreInitString "
             "FederateInfoSetFlagOption CreateValueFederate FederateRegisterGlobalPublication "
             "FederateRegisterSubscription FederateEnterExecutingMode FederateRequestTime "
             "PublicationPublishString FederateDisconnect FederateFree").split()
    for name in names:
        setattr(fake_h, "helics" + name, lambda *args: None)
    monkeypatch.setitem(sys.modules, "helics", fake_h)
    spec = importlib.util.spec_from_file_location("wp3_grid_fed", repo_root / "federates/grid_fed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = []
    class FakeGridPack:
        def __init__(self, net, topo, workdir, np_, hostfile, qlim=True):
            started.append(qlim)
            self.net, self.qlim = net, qlim
            self.ready, self.setup_s, self.np = {}, 0, np_
            self.last, self.iterations, self.ranks = {}, [], []
        def solve(self, pd, qd, vset):
            if backend == "gridpack-fails":
                raise RuntimeError("test MPI failure")
            voltage, it, ok = self.net.solve_pf(pd, qd, vset, qlim=self.qlim)
            self.last_qlim = self.net.q_limits_at(voltage, qd) if self.qlim else {}
            return voltage, it, ok
    monkeypatch.setattr(module, "server_available", lambda: True)
    monkeypatch.setattr(module, "GridPackMPISolver", FakeGridPack)
    expected = enforce is not False
    cfg = dict(outdir=str(tmp_path), topology=str(repo_root / "cases/ieee118/topology.json"),
               pmu_buses=[49], seed=1, grid_step=0.5, duration=1, solver="builtin" if backend == "builtin" else "gridpack",
               event=dict(type="avr_fault", t=0.5, bus=49, vset=1.12),
               load=dict(walk=0, amplitude=0, period=20))
    if enforce is not None:
        cfg["qlim"] = enforce
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(cfg))
    module.main(str(path))
    rows = list(csv.DictReader((tmp_path / "grid_steps.csv").open()))
    assert len(rows) == 3 and list(rows[0])[-1] == "q_limited"
    assert all(row["converged"] == "True" for row in rows)
    assert "49:max" not in rows[0]["q_limited"]
    assert ("49:max" in rows[1]["q_limited"]) is expected
    assert all(row["solver"] == ("gridpack" if backend == "gridpack" else "builtin") for row in rows)
    assert started == ([] if backend == "builtin" else [expected])
    truth = list(csv.DictReader((tmp_path / "grid_truth.csv").open()))
    assert (float(truth[1]["v49"]) < 1.12) is expected


@pytest.mark.parametrize("enforce", [True, False])
def test_web_request_argv(enforce, run_request_model):
    argv = run_request_model(case="ieee118", qlim=enforce).argv()
    assert ("--no-qlim" in argv) is (not enforce)
    assert parse_args(argv).qlim is enforce
