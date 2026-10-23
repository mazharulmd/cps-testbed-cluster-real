#!/usr/bin/env python3
"""Grid federate: power system simulation with GridPACK, in one of two modes.

quasi-steady-state (mode "qss"): every grid step (0.5 s by default) the federate
  1. applies control commands that have arrived over the NS-3 network (generator
     voltage setpoints),
  2. updates the loads (slow sinusoidal variation + small random walk, plus an
     optional scheduled event),
  3. re-solves the power flow with GridPACK (or the built-in Newton-Raphson solver),
  4. publishes the true voltage/current phasors seen by every PMU.

dynamic (mode "dynamic"): GridPACK's dynamic simulation integrates the generators, exciters
  and governors in steps of a few milliseconds; the federate advances it one PMU frame at a
  time, moves exciter references for commands and AVR events, and publishes the phasors of
  every frame. Faults and line or generator trips are scheduled in GridPACK. Loads are
  constant impedances.

Both modes log the true grid state so experiments can be scored afterwards.

Usage: grid_fed.py <run_dir>/run_config.json
"""
import csv, json, os, sys, time
import numpy as np
import helics as h

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpslib.network import Network
from cpslib.pmu import PmuLayout
from cpslib.gridpack import (GridPackMPISolver, GridPackDynamicSolver, server_available,
                             DSF_SERVER)


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    out = cfg["outdir"]
    topo = json.load(open(cfg["topology"]))
    net = Network(topo)
    layout = PmuLayout(net, cfg["pmu_buses"])
    rng = np.random.default_rng(cfg["seed"])
    dt, T = cfg["grid_step"], cfg["duration"]

    qlim = cfg.get("qlim", True)
    solver, dynamic = None, cfg.get("mode") == "dynamic"
    mpi = cfg.get("mpi") or {}
    ev = cfg.get("event") or {}
    if dynamic:
        if not server_available(DSF_SERVER):
            sys.exit("[GRID] dynamic mode needs GridPACK's dsf_server (CPS_DSF_SERVER)")
        solver = GridPackDynamicSolver(net, topo, os.path.join(out, "gridpack"), cfg.get("grid_dir"),
                                       T, dt, ev, mpi.get("np", 1), mpi.get("hostfile"),
                                       cfg.get("dyn_step", 0.005), qlim=qlim)
        print(f"[GRID] GridPACK dsf_server: {solver.ready.get('nranks')} MPI ranks, {solver.ready.get('ngen')} "
              f"machines ({solver.machine_info['source']}), integration step {1000 * solver.dt:.3f} ms, "
              f"setup {solver.setup_s:.2f} s", flush=True)
        json.dump(dict(solver.machine_info, dt=solver.dt), open(os.path.join(out, "dynamics.json"), "w"), indent=1)
    elif cfg["solver"] == "gridpack":
        if server_available():
            # one MPI job for the whole run, started before joining HELICS (partitioning takes a while)
            solver = GridPackMPISolver(net, topo, os.path.join(out, "gridpack"),
                                       mpi.get("np", 1), mpi.get("hostfile"), qlim=qlim)
            print(f"[GRID] GridPACK pf_server: {solver.ready.get('nranks')} MPI ranks, "
                  f"setup {solver.setup_s:.2f} s", flush=True)
        else:
            print("[GRID] GridPACK pf_server not found (CPS_PF_SERVER); using the built-in solver", flush=True)
    pd0 = np.array([b["pd"] for b in topo["buses"]])
    qd0 = np.array([b["qd"] for b in topo["buses"]])
    vset = {}
    for g in topo["gens"]:
        if g["status"]:
            vset[g["bus"]] = g["vg"]
    lp = cfg["load"]
    walk = np.zeros(net.n)

    fi = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreTypeFromString(fi, "zmq")
    h.helicsFederateInfoSetCoreInitString(fi, cfg.get("helics_core_init", "--federates=1"))
    h.helicsFederateInfoSetFlagOption(fi, h.HELICS_FLAG_UNINTERRUPTIBLE, True)
    fed = h.helicsCreateValueFederate("grid", fi)
    pub = h.helicsFederateRegisterGlobalPublication(fed, "grid/meas", h.HELICS_DATA_TYPE_STRING, "")
    # true state for observers (the live dashboard); no federate in the loop depends on it
    pub_status = h.helicsFederateRegisterGlobalPublication(fed, "grid/status", h.HELICS_DATA_TYPE_STRING, "")
    sub = h.helicsFederateRegisterSubscription(fed, "ns3/cmd_delivered", "")
    h.helicsFederateEnterExecutingMode(fed)
    print(f"[GRID] {len(topo['buses'])}-bus system, {len(cfg['pmu_buses'])} PMUs, solver={cfg['solver']}", flush=True)

    truth = open(os.path.join(out, "grid_truth.csv"), "w", newline="")
    tw = csv.writer(truth)
    tw.writerow(["t"] + [f"v{b}" for b in net.bus_ids])
    applied = open(os.path.join(out, "commands_applied.csv"), "w", newline="")
    aw = csv.writer(applied)
    aw.writerow(["t_applied", "gen_bus", "vset", "issued_t", "delivered_t", "reason"])
    steps = open(os.path.join(out, "grid_steps.csv"), "w", newline="")
    sw = csv.writer(steps)
    sw.writerow(["t", "solver", "converged", "iterations", "total_load_mw", "max_v", "max_v_bus", "min_v", "min_v_bus",
                 "solve_ms", "mpi_ranks", "gridpack_solve_ms", "q_limited"])
    if dynamic:
        dyn = open(os.path.join(out, "grid_dynamics.csv"), "w", newline="")
        dw = csv.writer(dyn)
        glabels = [f"{b}_{g}" for b, g in solver.gens]
        # generator frequencies (rotor speeds); bus frequencies come from voltage angle changes and
        # spike when the network switches (fault on/off), so they are logged but not summarised
        dw.writerow(["t", "f_min_hz", "f_min_gen", "f_max_hz", "f_max_gen", "f_coi_hz", "angle_spread_deg",
                     "bus_f_min_hz", "bus_f_max_hz"]
                    + [f"f_g{x}" for x in glabels] + [f"angle_g{x}" for x in glabels] + [f"p_g{x}" for x in glabels])
        f_nom = float(cfg.get("f_nominal", 60.0))

    seen_cmds = set()
    k = 0
    while True:
        t = round(k * dt, 9)
        if t > T + 1e-9:
            break
        h.helicsFederateRequestTime(fed, t)
        applied_now = []
        if h.helicsInputIsUpdated(sub):
            for c in json.loads(h.helicsInputGetString(sub) or "[]"):
                if c.get("id") in seen_cmds:
                    continue
                seen_cmds.add(c.get("id"))
                vset[c["gen_bus"]] = c["vset"]
                if dynamic:
                    solver.set_vset(c["gen_bus"], c["vset"])
                aw.writerow([t, c["gen_bus"], c["vset"], c.get("issued_t"), c.get("delivered_t"), c.get("reason", "")])
                applied_now.append({"gen_bus": c["gen_bus"], "vset": c["vset"], "issued_t": c.get("issued_t"),
                                    "delivered_t": c.get("delivered_t"), "reason": c.get("reason", "")})
                print(f"[GRID] t={t:.2f}s command applied: gen {c['gen_bus']} vset -> {c['vset']:.3f} pu", flush=True)
        # loads
        walk = walk + lp["walk"] * np.sqrt(dt) * rng.standard_normal(net.n)
        scale = 1 + lp["amplitude"] * np.sin(2 * np.pi * t / lp["period"]) + walk
        pd, qd = (pd0.copy(), qd0.copy()) if dynamic else (pd0 * scale, qd0 * scale)   # dynamic: constant-impedance loads
        if ev.get("type") == "load_step" and t >= ev["t"]:
            i = net.idx[ev["bus"]]
            pd[i] *= 1 + ev["pct"] / 100; qd[i] *= 1 + ev["pct"] / 100
        if ev.get("type") == "avr_fault" and ev.get("t") <= t and not ev.get("_done"):
            vset[ev["bus"]] = ev["vset"]; ev["_done"] = True
            if dynamic:
                solver.set_vset(ev["bus"], ev["vset"])
            aw.writerow([t, ev["bus"], ev["vset"], "", "", "event: AVR setpoint fault"])
            applied_now.append({"gen_bus": ev["bus"], "vset": ev["vset"], "reason": "event: AVR setpoint fault"})
            print(f"[GRID] t={t:.2f}s EVENT: AVR fault at gen {ev['bus']}, vset -> {ev['vset']}", flush=True)
        if dynamic and ev.get("type") in ("bus_fault", "line_trip", "gen_trip") and ev["t"] <= t and not ev.get("_done"):
            ev["_done"] = True     # applied inside GridPACK at its scheduled time; logged here
            aw.writerow([ev["t"], ev.get("bus", ev.get("from", "")), "", "", "", f"event: {ev['type'].replace('_', ' ')}"])
            applied_now.append({"gen_bus": ev.get("bus", ev.get("from")), "reason": f"event: {ev['type'].replace('_', ' ')}"})
            print(f"[GRID] t={t:.2f}s EVENT: {ev['type']} {ev}", flush=True)
        # solve
        V, it, ok, used = None, 0, False, cfg["solver"]
        t_solve = time.time()
        if dynamic:
            V, freq, gens, ok = solver.step(t)
            used = "gridpack-dynamic"
            if not ok:
                sys.exit(f"[GRID] t={t:.3f}s the dynamic simulation failed; see gridpack/dsf_server.log")
        elif solver:
            try:
                V, it, ok = solver.solve(pd, qd, vset)
            except RuntimeError as e:
                # the MPI job died (e.g. a node went down): finish the run with the built-in solver
                print(f"[GRID] t={t:.2f}s GridPACK failed ({e}); using the built-in solver from now on", flush=True)
                solver = None
        t_solve = time.time() - t_solve
        srv = getattr(solver, "last", None) or {}
        if not dynamic and (V is None or not ok):
            used = "builtin"
            V, it, ok = net.solve_pf(pd, qd, vset, qlim=qlim)
        # Dynamic exciters do not use the QSS PV/PQ switching mechanism.
        limited = ({} if dynamic else net.last_qlim if used == "builtin" else solver.last_qlim)
        vm = np.abs(V)
        tw.writerow([t] + [f"{x:.6f}" for x in vm])
        status = {"t": t, "vm": [round(float(x), 5) for x in vm], "solver": used, "applied": applied_now}
        # what GridPACK did in this step, for the dashboard's Inside page
        on_gp = used.startswith("gridpack")
        gp = {"server": ("dsf_server" if dynamic else "pf_server") if on_gp else "built-in Newton-Raphson",
              "ranks": getattr(solver, "np", 1) if on_gp else 1, "converged": bool(ok),
              "t_step_ms": round(1000 * t_solve, 2),
              "t_mpi_ms": round(1000 * float(srv["t_solve"]), 2) if on_gp and "t_solve" in srv else None,
              "t_apply_ms": round(1000 * float(srv["t_apply"]), 3) if on_gp and "t_apply" in srv else None,
              "load_mw": round(float(pd.sum()), 1), "max_v": [round(float(vm.max()), 4), net.bus_ids[int(vm.argmax())]],
              "min_v": [round(float(vm.min()), 4), net.bus_ids[int(vm.argmin())]], "buses": net.n}
        if dynamic:
            gp.update(steps=solver.steps, dt_ms=round(1000 * solver.dt, 3), machines=len(solver.gens))
        elif on_gp:
            # GridPACK prints the starting mismatch twice; keep one value per Newton iteration
            seq = []
            for _, tol in solver.iterations:
                if not seq or abs(tol - seq[-1]) > 1e-12 * max(abs(tol), 1):
                    seq.append(tol)
            gp["iterations"] = [[i, float(f"{tol:.3g}")] for i, tol in enumerate(seq)]
        else:
            gp["iterations"] = [[i, None] for i in range(1, int(it) + 1)]
        if on_gp and k % 30 == 0:
            gp["rank_map"] = solver.ranks
        status["gp"] = gp
        if dynamic:
            fg = [f_nom * g["speed"] for g in gens]
            # centre-of-inertia frequency of the machines in service
            hw = [solver.inertia.get((g["bus"], str(g["id"])), 1.0) if g["online"] else 0.0 for g in gens]
            coi = float(np.average(fg, weights=hw)) if sum(hw) > 0 else f_nom
            on = [g["angle"] for g in gens if g["online"]]
            spread = (max(on) - min(on)) if on else 0.0
            fb = freq if np.any(freq) else np.full(net.n, coi)
            live = [(x, g["bus"]) for x, g in zip(fg, gens) if g["online"]] or [(coi, "")]
            lo, hi = min(live), max(live)
            dw.writerow([t, f"{lo[0]:.5f}", lo[1], f"{hi[0]:.5f}", hi[1], f"{coi:.5f}", f"{spread:.3f}",
                         f"{fb.min():.5f}", f"{fb.max():.5f}"]
                        + [f"{x:.5f}" for x in fg] + [f"{g['angle']:.3f}" for g in gens] + [f"{g['p']:.4f}" for g in gens])
            status.update(freq_coi=round(coi, 5), freq_min=round(lo[0], 5), freq_max=round(hi[0], 5),
                          angle_spread=round(spread, 3),
                          gens=[{"bus": g["bus"], "f": round(x, 5), "angle": round(g["angle"], 3), "online": g["online"]}
                                for g, x in zip(gens, fg)])
        sw.writerow([t, used, ok, it, f"{pd.sum():.2f}", f"{vm.max():.5f}", net.bus_ids[int(vm.argmax())],
                     f"{vm.min():.5f}", net.bus_ids[int(vm.argmin())], f"{1000 * t_solve:.2f}",
                     getattr(solver, "np", 1) if used.startswith("gridpack") else "",
                     f"{1000 * float(srv['t_solve']):.2f}" if used.startswith("gridpack") and "t_solve" in srv else "",
                     " ".join(f"{bus}:{side}" for bus, side in sorted(limited.items()))])
        meas = {str(pm["bus"]): ph for pm, ph in zip(layout.pmus, layout.measure(V))}
        h.helicsPublicationPublishString(pub, json.dumps({"t": t, "pmus": meas}))
        h.helicsPublicationPublishString(pub_status, json.dumps(status))
        k += 1

    for f in (truth, applied, steps) + ((dyn,) if dynamic else ()):
        f.close()
    if hasattr(solver, "close"):
        solver.close()
    h.helicsFederateDisconnect(fed)
    h.helicsFederateFree(fed)
    print("[GRID] finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
