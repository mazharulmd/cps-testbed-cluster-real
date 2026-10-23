"""GridPACK for the grid federate: long-lived MPI jobs that the federate drives over
stdin/stdout, pf_server (power flow, gridpack/pf_server) and dsf_server (dynamic simulation,
gridpack/dsf_server)."""
import math, os, re, shutil, subprocess
import numpy as np

from .rawio import write_raw
from .dynamics import machine_data


# ---------------------------------------------------------------- persistent MPI server
PF_SERVER = os.environ.get("CPS_PF_SERVER", "/opt/cps/gridpack/bin/pf_server")
# This GridPACK build reads qlim as int (PFAppModule::p_qlim), so use 1/0.
SERVER_XML = """<?xml version="1.0" encoding="utf-8"?>
<Configuration>
  <Powerflow>
    <networkConfiguration> {raw} </networkConfiguration>
    <maxIteration>50</maxIteration>
    <tolerance>1.0e-6</tolerance>
    <qlim>{qlim}</qlim>
    <LinearSolver>
      <PETScOptions>
        -ksp_type preonly
        -pc_type lu
        -pc_factor_mat_solver_type mumps
      </PETScOptions>
    </LinearSolver>
  </Powerflow>
</Configuration>
"""


def server_available(path=PF_SERVER):
    return os.path.exists(path) and shutil.which("mpirun") is not None


# GridPACK's own progress lines: Newton iterations of the power flow and integration steps
ITERATION = re.compile(r"Iteration (\d+)[^T]*Tol:\s*([-+0-9.eE]+)")


class _MPIServer:
    """An MPI job started once; rank 0 reads commands on stdin and answers with "@@" lines.

    Besides the answers it keeps what GridPACK reports while it works, for the live view:
    ranks (host, buses and branches of every MPI rank), iterations (Newton iterations of
    the last power flow as (iteration, mismatch)) and steps (integration steps of the last
    dynamic step)."""

    def _start(self, server, xml, np_, hostfile, log):
        self.ranks, self.iterations, self.steps = [], [], 0
        cmd = ["mpirun", "-np", str(int(np_)), "--wdir", self.dir,
               "-x", "LD_LIBRARY_PATH", "-x", "PATH"]
        if hostfile:
            cmd += ["--hostfile", hostfile]
        cmd += [server, xml]
        self.np, self.hostfile = int(np_), hostfile
        self.log = open(log, "w")
        self.log.write(" ".join(cmd) + "\n"); self.log.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.ready = self._read_until("@@READY")
        self.setup_s = float(self.ready.get("t_setup", 0))
        self.last = {}

    def _read_until(self, tag, collect=None):
        for line in self.proc.stdout:
            if not line.startswith("@@"):
                self.log.write(line)
                if line.startswith("Time ="):
                    self.steps += 1
                else:
                    m = ITERATION.search(line)
                    if m:
                        self.iterations.append((int(m.group(1)), float(m.group(2))))
                continue
            parts = line.split()
            if parts[0] == "@@RANK" and len(parts) >= 5:
                self.ranks.append({"rank": int(parts[1]), "host": parts[2], "buses": int(parts[3]),
                                   "branches": int(parts[4])})
                continue
            if collect is not None and parts[0] in collect:
                collect[parts[0]].append(parts[1:])
            elif parts[0] == "@@WARN":
                self.log.write(line)
            elif parts[0] == tag:
                self.log.flush()
                return dict(p.split("=", 1) for p in parts[1:] if "=" in p)
        self.log.flush()
        raise RuntimeError(f"{os.path.basename(self.server)} exited (code {self.proc.wait()}); see {self.log.name}")

    def _send(self, lines):
        self.iterations, self.steps = [], 0
        self.proc.stdin.write("\n".join(lines) + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            try:
                self._send(["QUIT"])
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.log.close()


class GridPackMPISolver(_MPIServer):
    """Runs GridPACK as one long-lived MPI job (pf_server) spread over `np` ranks, which may
    sit on other cluster nodes (`hostfile`). The network is read and partitioned once; each
    solve() only sends the changed loads and generator setpoints to rank 0's stdin."""

    def __init__(self, net, topo, workdir, np_=1, hostfile=None, log=None, qlim=True):
        self.qlim, self.last_qlim = qlim, {}
        self.net, self.topo, self.dir = net, topo, os.path.abspath(workdir)
        os.makedirs(self.dir, exist_ok=True)
        raw = os.path.join(self.dir, "case.raw")
        write_raw(raw, topo)
        xml = os.path.join(self.dir, "input.xml")
        with open(xml, "w") as fh:
            fh.write(SERVER_XML.format(raw=raw, qlim=int(qlim)))
        # GridPACK only creates load records for buses with a non-zero load in the .raw file
        self.load_buses = [i for i, b in enumerate(topo["buses"]) if b["pd"] != 0 or b["qd"] != 0]
        # generator ids are numbered per bus in file order, as rawio writes them
        self.gens, count = [], {}
        for g in topo["gens"]:
            count[g["bus"]] = count.get(g["bus"], 0) + 1
            if g["status"]:
                self.gens.append((g["bus"], count[g["bus"]], g["vg"]))
        self.server = PF_SERVER
        self._start(PF_SERVER, xml, np_, hostfile, log or os.path.join(self.dir, "pf_server.log"))

    def solve(self, pd, qd, vset):
        lines = [f"LOAD {self.topo['buses'][i]['id']} {pd[i]:.6f} {qd[i]:.6f}" for i in self.load_buses]
        lines += [f"VSET {bus} {gid} {vset.get(bus, vg):.6f}" for bus, gid, vg in self.gens]
        self._send(lines + ["SOLVE"])
        # the server answers with @@RESULT, one @@V line per bus, then @@END
        self.last = head = self._read_until("@@RESULT")
        rows = {"@@V": []}
        self._read_until("@@END", collect=rows)
        V = np.zeros(self.net.n, complex)
        for bus, vm, va in rows["@@V"]:
            V[self.net.idx[int(bus)]] = float(vm) * np.exp(1j * np.deg2rad(float(va)))
        ok = head.get("ok") == "1" and len(rows["@@V"]) == self.net.n
        self.last_qlim = self.net.q_limits_at(V, qd) if self.qlim and ok else {}
        return V, 0, ok


# ---------------------------------------------------------------- dynamic simulation
DSF_SERVER = os.environ.get("CPS_DSF_SERVER", "/opt/cps/gridpack/bin/dsf_server")
DSF_XML = """<?xml version="1.0" encoding="utf-8"?>
<Configuration>
  <Powerflow>
    <networkConfiguration> {raw} </networkConfiguration>
    <maxIteration>50</maxIteration>
    <tolerance>1.0e-8</tolerance>
    <qlim>{qlim}</qlim>
    <LinearSolver>
      <PETScOptions>
        -ksp_type preonly
        -pc_type lu
        -pc_factor_mat_solver_type mumps
      </PETScOptions>
    </LinearSolver>
  </Powerflow>
  <Dynamic_simulation>
    <generatorParameters> {dyr} </generatorParameters>
    <simulationTime> {T} </simulationTime>
    <timeStep> {dt} </timeStep>
    <Events>
{events}
    </Events>
    <observations>
{observations}
    </observations>
    <LinearSolver>
      <PETScOptions>
        -ksp_type richardson
        -pc_type lu
        -pc_factor_mat_solver_type mumps
        -ksp_max_it 1
      </PETScOptions>
    </LinearSolver>
  </Dynamic_simulation>
</Configuration>
"""


def dsf_events(event, topo, dt):
    """Scheduled grid events in GridPACK's input format; times are put on the integration grid
    (GridPACK matches them to the step times)."""
    on_grid = lambda t: round(round(t / dt) * dt, 9)
    kind = event.get("type")
    if kind == "bus_fault":
        t = on_grid(event["t"])
        return [f"      <BusFault><begin>{t}</begin><end>{on_grid(t + event['duration'])}</end>"
                f"<bus>{event['bus']}</bus><yfault>0.0 99999.0</yfault></BusFault>"]
    if kind == "line_trip":
        out, count = [], {}
        for r in topo["branches"]:
            key = (min(r["from"], r["to"]), max(r["from"], r["to"]))
            count[key] = count.get(key, 0) + 1
            if {r["from"], r["to"]} == {event["from"], event["to"]} and r["status"]:
                out.append(f"      <LineStatus><time>{on_grid(event['t'])}</time><line>{r['from']} {r['to']}</line>"
                           f"<id>{count[key]}</id><status>0</status></LineStatus>")
        return out
    if kind == "gen_trip":
        return [f"      <GenStatus><time>{on_grid(event['t'])}</time><bus>{event['bus']}</bus>"
                f"<id>1</id><status>0</status></GenStatus>"]
    return []


class GridPackDynamicSolver(_MPIServer):
    """GridPACK's dynamic simulation (generators, exciters, governors) as one long-lived MPI job
    (dsf_server). Every step() integrates up to the requested time in steps of `dt` and returns
    the bus voltages and frequencies and the state of every generator. Generator voltage
    setpoints move the exciters' voltage references by the same amount."""

    def __init__(self, net, topo, workdir, grid_dir, duration, step, event=None, np_=1,
                 hostfile=None, max_dt=0.005, log=None, qlim=True):
        self.net, self.topo, self.dir = net, topo, os.path.abspath(workdir)
        os.makedirs(self.dir, exist_ok=True)
        # integration step: the largest that fits a whole number of times into a grid step
        self.dt = step / max(1, math.ceil(step / max_dt - 1e-9))
        dyr, machines, self.machine_info = machine_data(topo, grid_dir)
        raw, dyr_path = os.path.join(self.dir, "case.raw"), os.path.join(self.dir, "case.dyr")
        write_raw(raw, topo, machines=machines)
        with open(dyr_path, "w") as fh:
            fh.write(dyr)
        self.gens = sorted(machines, key=lambda k: (k[0], int(k[1])))
        # stored kinetic energy H * MVA of every machine, the weights of the centre-of-inertia frequency
        self.inertia = {k: m["h"] * m["mbase"] for k, m in machines.items()}
        obs = [f"      <observation><type>bus</type><busID>{b['id']}</busID></observation>\n"
               f"      <observation><type>busfrequency</type><busID>{b['id']}</busID></observation>"
               for b in topo["buses"]]
        obs += [f"      <observation><type>generator</type><busID>{b}</busID><generatorID>{g}</generatorID></observation>"
                for b, g in self.gens]
        xml = os.path.join(self.dir, "input.xml")
        with open(xml, "w") as fh:
            fh.write(DSF_XML.format(raw=raw, qlim=int(qlim), dyr=dyr_path, T=duration + 1.0,
                                    dt=f"{self.dt:.10g}",
                                    events="\n".join(dsf_events(event or {}, topo, self.dt)),
                                    observations="\n".join(obs)))
        self.vg0 = {}
        for g in topo["gens"]:
            if g["status"]:
                self.vg0.setdefault(g["bus"], g["vg"])
        self.pending = []
        # GridPACK reports the bus voltages after the first integration step; at t = 0 the
        # state is the initial power flow, which the testbed's own solver reproduces
        self.V0, _, _ = net.solve_pf(qlim=qlim)
        self.server = DSF_SERVER
        self._start(DSF_SERVER, xml, np_, hostfile, log or os.path.join(self.dir, "dsf_server.log"))

    def set_vset(self, bus, vset):
        """New terminal voltage setpoint of the generators at `bus` (exciter reference moved by
        the change from the initial setpoint)."""
        for b, gid in self.gens:
            if b == bus:
                self.pending.append(f"DVREF {b} {gid} {vset - self.vg0.get(b, 1.0):.6f}")

    def step(self, t):
        self._send(self.pending + [f"STEP {t:.10g}"])
        self.pending = []
        self.last = head = self._read_until("@@RESULT")
        rows = {"@@V": [], "@@G": []}
        self._read_until("@@END", collect=rows)
        V = np.zeros(self.net.n, complex)
        freq = np.zeros(self.net.n)
        for bus, vm, va, f in rows["@@V"]:
            i = self.net.idx[int(bus)]
            V[i] = float(vm) * np.exp(1j * np.deg2rad(float(va)))
            freq[i] = float(f)
        gens = [{"bus": int(b), "id": g, "speed": float(w), "angle": float(a), "p": float(p),
                 "q": float(q), "online": float(on) > 0.5} for b, g, w, a, p, q, on in rows["@@G"]]
        if t <= 1e-12 and not np.all(np.abs(V) > 1e-9):
            V = self.V0.copy()
        ok = len(rows["@@V"]) == self.net.n and np.all(np.isfinite(V))
        return V, freq, gens, ok
