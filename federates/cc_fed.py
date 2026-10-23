#!/usr/bin/env python3
"""Control-center federate: state estimation, bad data detection and voltage control.

For every time-aligned measurement set released by the PDC (via NS-3) it
  1. runs linear PMU state estimation (with chi-square bad data detection and
     largest-normalized-residual removal when enabled),
  2. checks the estimated bus voltages against the limits,
  3. sends a generator voltage setpoint command for the generator closest to a
     violating bus. Commands travel back through the NS-3 network.

Usage: cc_fed.py <run_dir>/run_config.json
"""
import csv, json, os, sys, time
import numpy as np
import helics as h

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpslib.network import Network
from cpslib.pmu import PmuLayout
from cpslib.estimation import StateEstimator
from cpslib.control import nearest_generators


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    out = cfg["outdir"]
    net = Network(cfg["topology"])
    layout = PmuLayout(net, cfg["pmu_buses"])
    se = StateEstimator(layout, sigma=cfg["se_sigma"], alpha=cfg["bdd_alpha"])
    col = {pm["bus"]: p for p, pm in enumerate(layout.pmus)}
    ctl = cfg["control"]
    vset = {}
    topo = json.load(open(cfg["topology"]))
    for g in topo["gens"]:
        if g["status"]:
            vset[g["bus"]] = g["vg"]
    nearest = nearest_generators(net, set(vset))
    last_cmd = {}
    cmd_id = 0

    fi = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreTypeFromString(fi, "zmq")
    h.helicsFederateInfoSetCoreInitString(fi, cfg.get("helics_core_init", "--federates=1"))
    fed = h.helicsCreateValueFederate("control_center", fi)
    pub = h.helicsFederateRegisterGlobalPublication(fed, "cc/commands", h.HELICS_DATA_TYPE_STRING, "")
    # what the control center sees, for observers (the live dashboard); nothing in the loop subscribes
    pub_status = h.helicsFederateRegisterGlobalPublication(fed, "cc/status", h.HELICS_DATA_TYPE_STRING, "")
    est_every, last_est_t = cfg.get("grid_step", 0.5), -1e9
    sub = h.helicsFederateRegisterSubscription(fed, "ns3/pdc", "")
    h.helicsFederateEnterExecutingMode(fed)
    print(f"[CC] state estimation over {len(layout.pmus)} PMUs, BDD={'on' if cfg['bdd'] else 'off'}, "
          f"control={'on' if ctl['enabled'] else 'off'}", flush=True)

    log = open(os.path.join(out, "cc_log.csv"), "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["t", "release_t", "helics_t", "pmus_received", "pmus_expected", "m", "observable", "J", "threshold",
                 "alarm", "first_J", "first_alarm", "removed_pmus", "max_v_est", "max_v_bus", "min_v_est", "min_v_bus",
                 "violations", "commands", "first_threshold"])
    est = open(os.path.join(out, "cc_estimates.csv"), "w", newline="")
    ew = csv.writer(est)
    ew.writerow(["t"] + [f"v{b}" for b in net.bus_ids])
    # the bus voltage magnitude each PMU delivered (as received, attacks included), blank when missing
    rcv = open(os.path.join(out, "cc_received.csv"), "w", newline="")
    rw = csv.writer(rcv)
    rw.writerow(["t"] + [f"v{pm['bus']}" for pm in layout.pmus])

    T = cfg["duration"]
    t = 0.0
    while t < T:
        t = h.helicsFederateRequestTime(fed, T)
        if not h.helicsInputIsUpdated(sub):
            continue
        cmds, sets, vm_send = [], [], None
        for s in json.loads(h.helicsInputGetString(sub)):
            meas = {}
            for bus, d in s["pmus"].items():
                p = col[int(bus)]
                for c, (re, im) in enumerate(d["ph"]):
                    meas[(p, c)] = complex(re, im)
            if not meas:
                continue
            t_est0, n_cmds0 = time.perf_counter(), len(cmds)
            if cfg["bdd"]:
                r = se.estimate_with_bdd(meas)
            else:
                r = se.estimate(meas)
                r.update(removed_pmus=[], first_J=r["J"], first_alarm=r["alarm"],
                         first_threshold=r["threshold"], first_observable=r["observable"])
            est_ms = 1000 * (time.perf_counter() - t_est0)
            vm = np.abs(r["V"])
            viol = [(net.bus_ids[i], float(vm[i])) for i in range(net.n)
                    if vm[i] > ctl["vmax"] or vm[i] < ctl["vmin"]]
            # act only when the violation is clearly beyond measurement noise
            db = ctl.get("deadband", 0.0)
            act = [(b, v) for b, v in viol if v > ctl["vmax"] + db or v < ctl["vmin"] - db]
            # hold control while bad data could not be cleaned out
            trusted = not r["alarm"]
            if ctl["enabled"] and trusted:
                for bus, v in sorted(act, key=lambda x: -abs(x[1] - 1)):
                    g = nearest.get(bus)
                    if g is None or t - last_cmd.get(g, -1e9) < ctl["cooldown"]:
                        continue
                    step = -ctl["step"] if v > ctl["vmax"] else ctl["step"]
                    new = float(np.clip(vset[g] + step, ctl["vset_min"], ctl["vset_max"]))
                    if abs(new - vset[g]) < 1e-9:
                        continue
                    vset[g] = new
                    last_cmd[g] = t
                    cmd_id += 1
                    cmds.append({"id": cmd_id, "gen_bus": g, "vset": round(new, 4), "issued_t": round(t, 6),
                                 "reason": f"V{bus}={v:.3f}"})
            lw.writerow([f"{s['t']:.6f}", f"{s['release_t']:.6f}", f"{t:.6f}", len(s["pmus"]), s["expected"],
                         r["m"], int(r["observable"]), f"{r['J']:.3f}", f"{r['threshold']:.3f}", int(r["alarm"]),
                         f"{r['first_J']:.3f}", int(r["first_alarm"]), " ".join(str(layout.pmus[p]["bus"]) for p in r["removed_pmus"]),
                         f"{vm.max():.5f}", net.bus_ids[int(vm.argmax())], f"{vm.min():.5f}",
                         net.bus_ids[int(vm.argmin())], len(viol), len(cmds), f"{r['first_threshold']:.3f}"])
            ew.writerow([f"{s['t']:.6f}"] + [f"{x:.5f}" for x in vm])
            rx = {int(b): abs(complex(*d["ph"][0])) for b, d in s["pmus"].items() if d["ph"]}
            rw.writerow([f"{s['t']:.6f}"] + [f"{rx[pm['bus']]:.5f}" if pm["bus"] in rx else "" for pm in layout.pmus])
            sets.append({"t": round(s["t"], 6), "J": round(float(r["first_J"]), 3), "thr": round(float(r["first_threshold"]), 3),
                         "alarm": int(r["first_alarm"]), "complete": round(len(s["pmus"]) / max(s["expected"], 1), 4),
                         "removed": [layout.pmus[p]["bus"] for p in r["removed_pmus"]], "viol": len(viol),
                         "max_v": round(float(vm.max()), 5), "max_bus": net.bus_ids[int(vm.argmax())],
                         # for the Inside page: the size of the estimation and what it took
                         "m": int(r["m"]), "pmus": len(s["pmus"]), "expected": s["expected"], "obs": int(r["first_observable"]), "J_final": round(float(r["J"]), 3),
                         "est_ms": round(est_ms, 2), "lag_ms": round(1000 * (t - s["t"]), 1), "cmds": len(cmds) - n_cmds0})
            if s["t"] - last_est_t >= est_every - 1e-9:
                vm_send, last_est_t = (round(s["t"], 6), [round(float(x), 5) for x in vm],
                                       [[b, round(v, 5)] for b, v in sorted(rx.items())],
                                       [layout.pmus[p]["bus"] for p in r["removed_pmus"]]), s["t"]
        if sets:
            h.helicsPublicationPublishString(pub_status, json.dumps(
                {"sets": sets, "t_est": vm_send[0] if vm_send else None, "vm_est": vm_send[1] if vm_send else None,
                 # PMU voltages as received and the PMUs removed as bad data, for the voltage profile
                 "rx": vm_send[2] if vm_send else None, "removed": vm_send[3] if vm_send else None}))
        if cmds:
            h.helicsPublicationPublishString(pub, json.dumps(cmds))
            for c in cmds:
                print(f"[CC] t={t:.3f}s {c['reason']} -> command gen {c['gen_bus']} vset {c['vset']}", flush=True)
    log.close(); est.close(); rcv.close()
    h.helicsFederateDisconnect(fed)
    h.helicsFederateFree(fed)
    print("[CC] finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
