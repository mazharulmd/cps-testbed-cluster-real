#!/usr/bin/env python3
"""Run one CPS co-simulation experiment: GridPACK grid + NS-3 network + control center.

Example:
  run_experiment.py --case 118 --duration 10 --latency 10 --loss 0.01 \
      --attack fdi-stealthy --target 25 --fake 1.0 --attack-start 3 --attack-end 8

Results go to <out>/<run id>/: configs, federate logs, per-frame network records,
state estimation log, true grid state and summary.json.
"""
import argparse, csv, json, os, shutil, socket, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpslib.network import Network
from cpslib.placement import optimal_placement
from cpslib.pmu import PmuLayout
from cpslib.attack import plan_fdi
from cpslib.estimation import sigma_from_polar_noise
from cpslib import cluster

CASES_DIR = os.environ.get("CPS_CASES", os.path.join(HERE, "..", "cases"))
NS3_BIN = os.environ.get("CPS_NS3_BIN", "/opt/cps/ns-3/helicstest")
PY = sys.executable


def parse_args(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--case", default="14", help="IEEE case (14, 30, 39, 57, 118, 300) or a label for --grid")
    a.add_argument("--grid", default=None, help="directory of an uploaded grid (topology.json); overrides --case")
    a.add_argument("--mpi-np", type=int, default=None, help="GridPACK MPI ranks (default: cluster setting or 1)")
    a.add_argument("--helics-port", type=int, default=None, help="HELICS broker port (default: first free)")
    a.add_argument("--name", default="run")
    a.add_argument("--duration", type=float, default=10.0, help="simulated seconds")
    a.add_argument("--mode", default="qss", choices=["qss", "dynamic"],
                   help="qss = a power flow every grid step; dynamic = GridPACK dynamic simulation "
                        "(generators, exciters, governors) advanced every PMU frame")
    a.add_argument("--grid-step", type=float, default=None,
                   help="grid federate step (s); default 0.5 (qss) or one PMU frame (dynamic)")
    a.add_argument("--dyn-step", type=float, default=0.005, help="largest integration step of the dynamic simulation (s)")
    a.add_argument("--rate", type=float, default=30.0, help="PMU reporting rate (frames/s)")
    a.add_argument("--noise-mag", type=float, default=0.001,
                   help="PMU relative magnitude noise std-dev (0..0.05; default: 0.001)")
    a.add_argument("--noise-ang", type=float, default=0.001,
                   help="PMU angle noise std-dev in radians (0..0.05; default: 0.001)")
    a.add_argument("--se-sigma", type=float, default=None,
                   help="override relative rectangular per-component noise std-dev (>0); "
                        "default: sqrt((noise_mag^2 + noise_ang^2)/2)")
    a.add_argument("--solver", default="gridpack", choices=["gridpack", "builtin"])
    a.add_argument("--placement", type=int, default=1, choices=[1, 2],
                   help="1 = minimum PMUs for observability, 2 = every bus seen by two PMUs")
    a.add_argument("--latency", type=float, default=10.0, help="mean PMU->PDC link latency (ms)")
    a.add_argument("--latency-spread", type=float, default=0.5, help="links vary by +/- this fraction")
    a.add_argument("--jitter", type=float, default=2.0, help="per-frame random delay (ms)")
    a.add_argument("--loss", type=float, default=0.0, help="packet loss rate per link, 0-1")
    a.add_argument("--pdc-wait", type=float, default=20.0, help="PDC wait window (ms)")
    a.add_argument("--attack", default="none", choices=["none", "fdi-simple", "fdi-stealthy", "drop", "delay"])
    a.add_argument("--target", type=int, default=None, help="attacked bus (default: highest-voltage bus)")
    a.add_argument("--fake", type=float, default=None, help="voltage the attacker wants the CC to see (pu)")
    a.add_argument("--attack-start", type=float, default=3.0)
    a.add_argument("--attack-end", type=float, default=8.0)
    a.add_argument("--attack-delay", type=float, default=100.0, help="extra delay for the delay attack (ms)")
    a.add_argument("--bdd", default="on", choices=["on", "off"], help="chi-square bad data detection")
    a.add_argument("--control", default="on", choices=["on", "off"])
    a.add_argument("--vmin", type=float, default=0.94)
    a.add_argument("--vmax", type=float, default=1.08)
    a.add_argument("--event", default="none",
                   help="none | avr:<gen bus>:<vset>:<t> | load:<bus>:<percent>:<t> (qss) | "
                        "fault:<bus>:<duration s>:<t> | line:<from>:<to>:<t> | gen:<bus>:<t> (dynamic)")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--out", default=os.environ.get("CPS_RESULTS", os.path.join(HERE, "..", "results", "runs")))
    args = a.parse_args(argv)
    for name in ("noise_mag", "noise_ang"):
        if not 0 <= getattr(args, name) <= 0.05:
            a.error(f"--{name.replace('_', '-')} must be finite and within 0..0.05")
    if args.se_sigma is not None and not (np.isfinite(args.se_sigma) and args.se_sigma > 0):
        a.error("--se-sigma must be finite and >0")
    return args


def parse_event(text):
    """--event value -> event dict (see --help)."""
    if text in (None, "", "none"):
        return {"type": "none"}
    f = text.split(":")
    try:
        if f[0] == "avr":
            return {"type": "avr_fault", "bus": int(f[1]), "vset": float(f[2]), "t": float(f[3])}
        if f[0] == "load":
            return {"type": "load_step", "bus": int(f[1]), "pct": float(f[2]), "t": float(f[3])}
        if f[0] == "fault":
            return {"type": "bus_fault", "bus": int(f[1]), "duration": float(f[2]), "t": float(f[3])}
        if f[0] == "line":
            return {"type": "line_trip", "from": int(f[1]), "to": int(f[2]), "t": float(f[3])}
        if f[0] == "gen":
            return {"type": "gen_trip", "bus": int(f[1]), "t": float(f[2])}
    except (IndexError, ValueError):
        pass
    raise SystemExit(f"bad --event '{text}' (see --help)")


def case_dir(args):
    return os.path.abspath(args.grid or os.path.join(CASES_DIR, f"ieee{args.case}"))


def free_port(base):
    """First broker port base + 100k not in use here (brokers of parallel runs are spaced
    apart because each HELICS core takes a few ports above its broker's)."""
    for k in range(50):
        port = base + 100 * k
        with socket.socket() as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free HELICS port")


def build_configs(args, run_dir, cl, port):
    se_sigma = args.se_sigma if args.se_sigma is not None else sigma_from_polar_noise(
        args.noise_mag, args.noise_ang)
    topo_path = os.path.join(case_dir(args), "topology.json")
    topo = json.load(open(topo_path))
    net = Network(topo)
    rng = np.random.default_rng(args.seed)
    pmu_buses, _ = optimal_placement(net, args.placement)
    layout = PmuLayout(net, pmu_buses)
    V0, _, _ = net.solve_pf()

    event = parse_event(args.event)

    # the attacker plans against the operating point it will face (including the event)
    V_plan = V0
    if event["type"] == "avr_fault":
        V_plan, _, _ = net.solve_pf(vset={event["bus"]: event["vset"]})
    elif event["type"] == "load_step":
        pd = np.array([b["pd"] for b in topo["buses"]]); qd = np.array([b["qd"] for b in topo["buses"]])
        i = net.idx[event["bus"]]; pd[i] *= 1 + event["pct"] / 100; qd[i] *= 1 + event["pct"] / 100
        V_plan, _, _ = net.solve_pf(pd, qd)
    target = args.target or net.bus_ids[int(np.argmax(np.abs(V_plan)))]
    attack = {"type": "none"}
    plan = None
    if args.attack.startswith("fdi"):
        fake = args.fake if args.fake is not None else round(abs(V_plan[net.idx[target]]) - 0.04, 3)
        plan = plan_fdi(layout, V_plan, target, fake, mode=args.attack.split("-")[1])
        attack = {"type": "fdi", "targets": plan["compromised_pmus"], "deltas": plan["deltas"],
                  "ref": plan["ref"], "v_fake": fake}
    elif args.attack in ("drop", "delay"):
        # attack the PMUs that observe the target bus (its own and its neighbours')
        col = np.abs(layout.column(net.idx[target])) > 1e-9
        obs = sorted({layout.pmus[p]["bus"] for (p, c), r in layout.row.items() if col[r]})
        attack = {"type": args.attack, "targets": obs, "delay_ms": args.attack_delay}
    attack.update(start=args.attack_start, end=args.attack_end)

    def link_delay():
        return round(args.latency * rng.uniform(1 - args.latency_spread, 1 + args.latency_spread), 3)

    gens = sorted({g["bus"] for g in topo["gens"] if g["status"]})
    ns3 = {"outdir": run_dir, "duration": args.duration, "rate": args.rate, "seed": args.seed,
           "pdc_wait_ms": args.pdc_wait, "proc_ms": 2.0, "jitter_ms": args.jitter, "loss": args.loss,
           "access_rate": "10Mbps", "noise": {"mag": args.noise_mag, "ang": args.noise_ang}, "attack": attack,
           "pmus": [{"id": pm["id"], "bus": pm["bus"], "nch": len(pm["channels"]), "delay_ms": link_delay()}
                    for pm in layout.pmus],
           "gens": [{"bus": g, "delay_ms": link_delay()} for g in gens],
           "helics_core_init": cluster.core_init(cl, cl["placement"].get("ns3", cl["head"]), port)}
    run = {"outdir": run_dir, "topology": topo_path, "grid_dir": case_dir(args), "case": args.case,
           "duration": args.duration, "mode": args.mode, "dyn_step": args.dyn_step, "grid_step": args.grid_step,
           "seed": args.seed, "solver": args.solver, "pmu_buses": pmu_buses,
           "load": {"amplitude": 0.02, "period": 20.0, "walk": 0.002}, "event": event,
           "se_sigma": se_sigma, "bdd_alpha": 0.01, "bdd": args.bdd == "on",
           "control": {"enabled": args.control == "on", "vmin": args.vmin, "vmax": args.vmax, "step": 0.01,
                       "deadband": 0.003, "cooldown": 1.0, "vset_min": 0.95, "vset_max": 1.10},
           "mpi": {"np": args.mpi_np, "hostfile": cl["mpi"].get("hostfile") if cl["nodes"] else None},
           "helics_port": port, "helics_core_init": cluster.core_init(cl, cl["placement"].get("grid", cl["head"]), port),
           "cc_core_init": cluster.core_init(cl, cl["placement"].get("cc", cl["head"]), port)}
    if args.mode == "dynamic" and event["type"] == "load_step":
        raise SystemExit("load steps are available in the quasi-steady-state mode only")
    if args.mode == "qss" and event["type"] in ("bus_fault", "line_trip", "gen_trip"):
        raise SystemExit(f"{event['type'].replace('_', ' ')} events need --mode dynamic")
    meta = {"args": vars(args), "target_bus": target, "attack_plan": plan, "grid_name": grid_name(args, topo),
            "cluster": {"placement": {r: cl["placement"].get(r, cl["head"]) for r in ("broker", "grid", "ns3", "cc")},
                        "mpi_np": args.mpi_np, "mpi_hosts": [n["host"] for n in cl["nodes"]], "helics_port": port},
            "pmus": layout.describe(), "n_buses": net.n, "n_branches": len(net.branches),
            "n_gens": len(gens), "base_case_max_v": float(np.abs(V0).max()),
            "control_center": {"se_sigma": se_sigma}}
    for name, obj in (("ns3_config.json", ns3), ("run_config.json", run), ("meta.json", meta)):
        with open(os.path.join(run_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1)
    return meta


def grid_name(args, topo):
    if args.grid:
        return topo.get("name") or os.path.basename(os.path.normpath(args.grid))
    return f"IEEE {args.case}-bus"


def run_federation(run_dir, cl, port):
    """Start the broker, the three federates and the observer on their nodes; stop all if one fails."""
    where = {r: cl["placement"].get(r, cl["head"]) for r in ("broker", "grid", "ns3", "cc")}
    where["observer"] = where["broker"]      # the live dashboard reads its output on the head
    cfg = os.path.join(run_dir, "run_config.json")
    cc_cfg = os.path.join(run_dir, "cc_config.json")
    run = json.load(open(cfg))
    json.dump(dict(run, helics_core_init=run["cc_core_init"]), open(cc_cfg, "w"), indent=1)
    obs_cfg = os.path.join(run_dir, "observer_config.json")
    json.dump(dict(run, helics_core_init=cluster.core_init(cl, where["observer"], run["helics_port"])),
              open(obs_cfg, "w"), indent=1)
    argv = {
        "broker": cluster.broker_args(port),
        "grid": [PY, os.path.join(HERE, "grid_fed.py"), cfg],
        "cc": [PY, os.path.join(HERE, "cc_fed.py"), cc_cfg],
        "ns3": [NS3_BIN, f"--config={os.path.join(run_dir, 'ns3_config.json')}"],
        "observer": [PY, os.path.join(HERE, "observer_fed.py"), obs_cfg],
    }
    logs = {n: open(os.path.join(run_dir, f"{n}.log"), "w") for n in argv}
    procs = {}
    for n in ("broker", "grid", "cc", "ns3", "observer"):
        procs[n] = cluster.launch(where[n], argv[n], logs[n], HERE)
        if n == "broker":
            time.sleep(0.5)
    t0 = time.time()
    codes = {}
    # if one federate fails the others would wait forever for it, so stop them all
    while len(codes) < len(procs):
        for n, p in procs.items():
            if n not in codes and p.poll() is not None:
                codes[n] = p.returncode
        failed = [n for n, c in codes.items() if c != 0]
        if failed or time.time() - t0 > 900:
            for n, p in procs.items():
                if n not in codes:
                    p.kill(); p.wait()
                    codes[n] = "killed" if failed else "timeout"
                    cluster.kill_remote(where[n], run_dir)
        time.sleep(0.2)
    for f in logs.values():
        f.close()
    return codes, time.time() - t0


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def analyze(run_dir, meta, codes, wall):
    args = meta["args"]
    s = {"name": args["name"], "case": meta["grid_name"], "exit_codes": codes, "wall_time_s": round(wall, 1),
         "cluster": meta["cluster"]}
    ns3 = json.load(open(os.path.join(run_dir, "ns3_summary.json")))
    s["network"] = ns3
    truth = read_csv(os.path.join(run_dir, "grid_truth.csv"))
    steps = read_csv(os.path.join(run_dir, "grid_steps.csv"))
    cc = read_csv(os.path.join(run_dir, "cc_log.csv"))
    est = read_csv(os.path.join(run_dir, "cc_estimates.csv"))
    applied = read_csv(os.path.join(run_dir, "commands_applied.csv"))
    vmax, vmin = args["vmax"], args["vmin"]
    tgt = meta["target_bus"]
    tt = np.array([float(r["t"]) for r in truth])

    def true_at(t, bus):
        i = int(np.searchsorted(tt, t + 1e-9) - 1)
        return float(truth[max(i, 0)][f"v{bus}"])

    viol_steps = [r for r in truth if any(not (vmin <= float(v) <= vmax) for k, v in r.items() if k != "t")]
    s["grid"] = {"steps": len(steps), "solver_used": sorted({r["solver"] for r in steps}),
                 "all_converged": all(r["converged"] == "True" for r in steps),
                 "true_violation_time_s": round(len(viol_steps) * args["grid_step"], 3),
                 "first_violation_t": float(viol_steps[0]["t"]) if viol_steps else None,
                 "last_violation_t": float(viol_steps[-1]["t"]) if viol_steps else None,
                 "commands_applied": len([r for r in applied if not r["reason"].startswith("event")])}
    gp = [float(r["gridpack_solve_ms"]) for r in steps if r.get("gridpack_solve_ms")]
    sv = [float(r["solve_ms"]) for r in steps if r.get("solve_ms")]
    dyn_path = os.path.join(run_dir, "grid_dynamics.csv")
    if os.path.exists(dyn_path):
        dyn = read_csv(dyn_path)
        info = json.load(open(os.path.join(run_dir, "dynamics.json"))) if os.path.exists(os.path.join(run_dir, "dynamics.json")) else {}
        fmin = min(dyn, key=lambda r: float(r["f_min_hz"])) if dyn else None
        fmax = max(dyn, key=lambda r: float(r["f_max_hz"])) if dyn else None
        s["dynamics"] = {
            "machine_data": info.get("source"), "machines": info.get("machines"),
            "integration_step_ms": round(1000 * info["dt"], 4) if info.get("dt") else None,
            "f_min_hz": round(float(fmin["f_min_hz"]), 4) if fmin else None,
            "f_min_t": float(fmin["t"]) if fmin else None, "f_min_gen": fmin["f_min_gen"] if fmin else None,
            "f_max_hz": round(float(fmax["f_max_hz"]), 4) if fmax else None,
            "f_final_hz": round(float(dyn[-1]["f_coi_hz"]), 4) if dyn else None,
            "angle_spread_initial_deg": round(float(dyn[0]["angle_spread_deg"]), 2) if dyn else None,
            "angle_spread_max_deg": round(max(float(r["angle_spread_deg"]) for r in dyn), 2) if dyn else None,
            "angle_spread_final_deg": round(float(dyn[-1]["angle_spread_deg"]), 2) if dyn else None}
    s["gridpack"] = {"mpi_ranks": args.get("mpi_np"),
                     "solve_ms_mean": round(float(np.mean(gp)), 2) if gp else None,
                     "solve_ms_max": round(float(np.max(gp)), 2) if gp else None,
                     "step_ms_mean": round(float(np.mean(sv)), 2) if sv else None}
    # state estimation accuracy (clean data) and detection
    errs, missed = [], 0
    ast, aend = args["attack_start"], args["attack_end"]
    in_atk = lambda t: args["attack"] != "none" and ast <= t < aend
    for r in est:
        t = float(r["t"])
        if in_atk(t):
            continue
        errs += [abs(float(r[k]) - true_at(t, k[1:])) for k in r if k != "t"]
    for r in cc:
        t = float(r["t"])
        true_viol = any(not (vmin <= float(v) <= vmax) for k, v in truth[max(int(np.searchsorted(tt, t + 1e-9) - 1), 0)].items() if k != "t")
        if true_viol and int(r["violations"]) == 0:
            missed += 1
    alarms_in = [int(r["first_alarm"]) for r in cc if in_atk(float(r["t"]))]
    alarms_out = [int(r["first_alarm"]) for r in cc if not in_atk(float(r["t"]))]
    first_alarm = next((float(r["t"]) for r in cc if in_atk(float(r["t"])) and r["first_alarm"] == "1"), None)
    lat = [float(r["release_t"]) - float(r["t"]) for r in cc]
    s["control_center"] = {
        "sets_processed": len(cc),
        "complete_sets_pct": round(100 * np.mean([r["pmus_received"] == r["pmus_expected"] for r in cc]), 1) if cc else 0,
        "observable_pct": round(100 * np.mean([r["observable"] == "1" for r in cc]), 1) if cc else 0,
        "se_mean_abs_err_pu": round(float(np.mean(errs)), 5) if errs else None,
        "pdc_release_latency_ms": {"mean": round(1000 * float(np.mean(lat)), 2), "max": round(1000 * float(np.max(lat)), 2)} if lat else None,
        "false_alarm_rate_pct": round(100 * np.mean(alarms_out), 2) if alarms_out else 0,
        "sets_with_true_violation_not_seen": missed,
    }
    atk = {"type": args["attack"]}
    if args["attack"] != "none":
        seen = [float(r[f"v{tgt}"]) for r in est if in_atk(float(r["t"]))]
        true = [true_at(float(r["t"]), tgt) for r in est if in_atk(float(r["t"]))]
        atk.update({"target_bus": tgt, "window_s": [ast, aend],
                    "compromised_pmus": meta["attack_plan"]["compromised_pmus"] if meta["attack_plan"] else
                    json.load(open(os.path.join(run_dir, "ns3_config.json")))["attack"]["targets"],
                    "detection_rate_pct": round(100 * np.mean(alarms_in), 1) if alarms_in else 0,
                    "time_to_detect_s": round(first_alarm - ast, 3) if first_alarm is not None else None,
                    "target_true_v_mean": round(float(np.mean(true)), 4) if true else None,
                    "target_seen_v_mean": round(float(np.mean(seen)), 4) if seen else None})
    s["attack"] = atk
    s["control_center"]["se_sigma"] = meta["control_center"]["se_sigma"]
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=2)
    return s


def main(argv=None):
    args = parse_args(argv)
    cl = cluster.load()
    if args.mpi_np is None:
        args.mpi_np = int(cl["mpi"].get("default_np", 1))
    if args.grid_step is None:
        args.grid_step = 1.0 / args.rate if args.mode == "dynamic" else 0.5
    port = args.helics_port or free_port(cl["helics_port_base"])
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{args.name}"
    run_dir = os.path.abspath(os.path.join(args.out, run_id))
    os.makedirs(run_dir, exist_ok=True)
    try:
        cl, down = cluster.for_run(cl, run_dir, args.mpi_np if args.solver == "gridpack" or args.mode == "dynamic" else 1)
    except RuntimeError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise SystemExit(f"[RUN] {run_id}: {e}")
    if down:
        print(f"[RUN] not reachable, left out of this run: {', '.join(down)}", flush=True)
    meta = build_configs(args, run_dir, cl, port)
    print(f"[RUN] {run_id}: {meta['grid_name']}, {args.mode}, {len(meta['pmus'])} PMUs, attack={args.attack}, "
          f"GridPACK MPI ranks={args.mpi_np}, HELICS port {port}", flush=True)
    codes, wall = run_federation(run_dir, cl, port)
    if any(c != 0 for c in codes.values()):
        print(f"[RUN] federate exit codes: {codes} (see *.log in {run_dir})", flush=True)
        # the federate that failed first says why; the others were stopped because of it
        for n, c in codes.items():
            if c not in (0, "killed", "timeout"):
                tail = open(os.path.join(run_dir, f"{n}.log")).read().strip().splitlines()[-6:]
                print(f"[RUN] {n} failed:\n  " + "\n  ".join(tail), flush=True)
        if not os.path.exists(os.path.join(run_dir, "ns3_summary.json")):
            raise SystemExit(f"[RUN] {run_id} did not finish; no results")
    s = analyze(run_dir, meta, codes, wall)
    print(json.dumps(s, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
