#!/usr/bin/env python3
"""Convert MATPOWER/PYPOWER IEEE test cases into GridPACK inputs.

For each case this writes, under cases/<name>/:
  <NAME>.raw       PSS/E v23 network file (same layout GridPACK reads for IEEE14)
  input.xml        GridPACK power-flow configuration
  topology.json    buses, branches (pi-model params) and generators, used by the
                   PMU placement, state estimation and NS-3 network builders
  reference.csv    reference power-flow solution (testbed NR solver + MATPOWER stored solution)

Run in an environment with pypower and numpy:
  python tools/make_cases.py            # all cases
  python tools/make_cases.py 118 300    # selected cases
"""
import importlib, json, os, sys
import numpy as np

CASES = {"14": "case14", "30": "case30", "39": "case39", "57": "case57",
         "118": "case118", "300": "case300"}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases")

# MATPOWER column indices
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE = range(11)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = range(11)

INPUT_XML = """<?xml version="1.0" encoding="utf-8"?>
<Configuration>
  <Powerflow>
    <networkConfiguration> {raw} </networkConfiguration>
    <maxIteration>50</maxIteration>
    <tolerance>1.0e-6</tolerance>
    <qlim>{qlim}</qlim>
    <outputFormat>csv</outputFormat>
    <outputFile>{out}</outputFile>
    <LinearSolver>
      <PETScOptions>
        -ksp_type richardson
        -pc_type lu
        -pc_factor_mat_solver_type superlu_dist
        -ksp_max_it 1
      </PETScOptions>
    </LinearSolver>
    <UseNewton>false</UseNewton>
    <NonlinearSolver>
      <SolutionTolerance>1.0E-08</SolutionTolerance>
      <FunctionTolerance>1.0E-08</FunctionTolerance>
      <MaxIterations>50</MaxIterations>
      <PETScOptions>
        -ksp_type richardson
        -pc_type lu
        -pc_factor_mat_solver_type superlu_dist
        -ksp_max_it 1
        -snes_type newtonls
        -snes_linesearch_type basic
      </PETScOptions>
    </NonlinearSolver>
  </Powerflow>
</Configuration>
"""


def write_raw(path, ppc, title):
    bus, gen, br = ppc["bus"], ppc["gen"], ppc["branch"]
    base = ppc["baseMVA"]
    L = [f"0  {base:.3f}", f" {title}", " generated from MATPOWER data by tools/make_cases.py"]
    for b in bus:
        kv = b[BASE_KV] if b[BASE_KV] > 0 else 100.0
        L.append(f"{int(b[BUS_I]):7d},{int(b[BUS_TYPE]):2d},{b[PD]:10.3f},{b[QD]:10.3f},{b[GS]:10.3f},"
                 f"{b[BS]:10.3f},{int(b[BUS_AREA]):4d},{b[VM]:8.5f},{b[VA]:9.4f},"
                 f"'BUS-{int(b[BUS_I]):<8d}',{kv:9.4f},{int(b[ZONE]):4d}")
    L.append("0 / END OF BUS DATA, BEGIN GENERATOR DATA")
    count = {}
    for g in gen:
        i = int(g[GEN_BUS]); count[i] = count.get(i, 0) + 1
        L.append(f"{i:7d},'{count[i]:<2d}',{g[PG]:10.3f},{g[QG]:10.3f},{g[QMAX]:10.3f},{g[QMIN]:10.3f},"
                 f"{g[VG]:8.5f},     0,{(g[MBASE] or base):10.3f},   0.00000,   1.00000,   0.00000,"
                 f"   0.00000,   1.00000,{int(g[GEN_STATUS] > 0)},  100.0,{g[PMAX]:10.3f},{g[PMIN]:10.3f}")
    L.append("0 / END OF GENERATOR DATA, BEGIN BRANCH DATA")
    ckt = {}
    for r in br:
        f, t = int(r[F_BUS]), int(r[T_BUS])
        key = (min(f, t), max(f, t)); ckt[key] = ckt.get(key, 0) + 1
        L.append(f"{f:7d},{t:7d},'{ckt[key]:<2d}',{r[BR_R]:10.5f},{r[BR_X]:10.5f},{r[BR_B]:10.5f},"
                 f"{r[RATE_A]:9.2f},{r[RATE_B]:9.2f},{r[RATE_C]:9.2f},{r[TAP]:8.5f},{r[SHIFT]:8.3f},"
                 f" 0.00000, 0.00000, 0.00000, 0.00000,{int(r[BR_STATUS] > 0)}")
    L.append("0 / END OF BRANCH DATA, BEGIN TRANSFORMER ADJUSTMENT DATA")
    L.append("0 / END OF TRANSFORMER ADJUSTMENT DATA, BEGIN AREA DATA")
    slack = int(bus[bus[:, BUS_TYPE] == 3][0, BUS_I])
    for a in sorted({int(x) for x in bus[:, BUS_AREA]}):
        L.append(f"{a:4d},{slack:7d},     0.0,  3.000,'AREA_{a:<7d}'")
    L += ["0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA",
          "0 / END OF TWO-TERMINAL DC DATA, BEGIN SWITCHED SHUNT DATA",
          "0 / END OF SWITCHED SHUNT DATA, BEGIN IMPEDANCE CORRECTION DATA",
          "0 / END OF IMPEDANCE CORRECTION DATA, BEGIN MULTI-TERMINAL DC DATA",
          "0 / END OF MULTI-TERMINAL DC DATA, BEGIN MULTI-SECTION LINE DATA",
          "0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA"]
    for z in sorted({int(x) for x in bus[:, ZONE]}):
        L.append(f"{z:5d},'ZONE_{z:<7d}'")
    L += ["0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA",
          "0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA",
          "    1,'OWNER_1     '",
          "0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA",
          "0 / END OF FACTS DEVICE DATA"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def topology(ppc):
    bus, gen, br = ppc["bus"], ppc["gen"], ppc["branch"]
    return {
        "baseMVA": float(ppc["baseMVA"]),
        "buses": [{"id": int(b[BUS_I]), "type": int(b[BUS_TYPE]), "pd": float(b[PD]), "qd": float(b[QD]),
                   "gs": float(b[GS]), "bs": float(b[BS]), "base_kv": float(b[BASE_KV]),
                   "area": int(b[BUS_AREA])} for b in bus],
        "gens": [{"bus": int(g[GEN_BUS]), "pg": float(g[PG]), "vg": float(g[VG]), "qmax": float(g[QMAX]),
                  "qmin": float(g[QMIN]), "status": int(g[GEN_STATUS] > 0)} for g in gen],
        "branches": [{"from": int(r[F_BUS]), "to": int(r[T_BUS]), "r": float(r[BR_R]), "x": float(r[BR_X]),
                      "b": float(r[BR_B]), "tap": float(r[TAP]), "shift": float(r[SHIFT]),
                      "status": int(r[BR_STATUS] > 0)} for r in br],
    }


def reference(ppc, topo, path):
    """Solve with the testbed's own Newton-Raphson solver and compare with the solved
    voltages stored in the MATPOWER case."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "federates"))
    from cpslib.network import Network
    net = Network(topo)
    # Historical references are unconstrained; keep them reproducible.
    V, it, ok = net.solve_pf(qlim=False)
    vm, va = np.abs(V), np.rad2deg(np.angle(V))
    va -= va[[b["type"] for b in net.buses].index(3)]
    bus = ppc["bus"]
    va_ref = bus[:, VA] - bus[bus[:, BUS_TYPE] == 3][0, VA]
    err_vm = np.max(np.abs(vm - bus[:, VM])); err_va = np.max(np.abs(va - va_ref))
    with open(path, "w") as fh:
        fh.write("bus_id,vm_pu,va_deg,matpower_vm_pu,matpower_va_deg\n")
        for i, b in enumerate(bus):
            fh.write(f"{int(b[BUS_I])},{vm[i]:.6f},{va[i]:.4f},{b[VM]:.6f},{va_ref[i]:.4f}\n")
    return ok, it, err_vm, err_va


def main(names):
    for n in names:
        ppc = getattr(importlib.import_module(f"pypower.{CASES[n]}"), CASES[n])()
        ppc["bus"], ppc["gen"], ppc["branch"] = (np.asarray(ppc[k], dtype=float) for k in ("bus", "gen", "branch"))
        d = os.path.join(ROOT, f"ieee{n}"); os.makedirs(d, exist_ok=True)
        raw = f"IEEE{n}.raw"
        write_raw(os.path.join(d, raw), ppc, f"IEEE {n}-bus test system")
        with open(os.path.join(d, "input.xml"), "w") as fh:
            fh.write(INPUT_XML.format(raw=raw, out=f"pf_IEEE{n}", qlim="1"))
        topo = topology(ppc)
        with open(os.path.join(d, "topology.json"), "w") as fh:
            json.dump(topo, fh, indent=1)
        ok, it, evm, eva = reference(ppc, topo, os.path.join(d, "reference.csv"))
        print(f"ieee{n}: {len(ppc['bus'])} buses, {len(ppc['branch'])} branches, {len(ppc['gen'])} gens | "
              f"NR converged={ok} in {it} it, max |dV| vs MATPOWER {evm:.2e} pu, max |dθ| {eva:.2e} deg")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
