"""Network model built from cases/<name>/topology.json (MATPOWER conventions)."""
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class Network:
    def __init__(self, topo):
        if isinstance(topo, str):
            with open(topo) as fh:
                topo = json.load(fh)
        self.base = topo["baseMVA"]
        self.buses = topo["buses"]
        self.gens = topo["gens"]
        self.branches = [b for b in topo["branches"] if b["status"]]
        self.bus_ids = [b["id"] for b in self.buses]
        self.idx = {b: i for i, b in enumerate(self.bus_ids)}
        self.n = len(self.bus_ids)
        self.last_qlim = {}
        self._build_admittances()

    def _build_admittances(self):
        """Branch admittances in MATPOWER form: If = Yf V, It = Yt V, Ybus = Cf'Yf + Ct'Yt + diag(Ysh)."""
        nl, n = len(self.branches), self.n
        Ytt = np.zeros(nl, complex); Yff = np.zeros(nl, complex)
        Yft = np.zeros(nl, complex); Ytf = np.zeros(nl, complex)
        f = np.zeros(nl, int); t = np.zeros(nl, int)
        for k, br in enumerate(self.branches):
            ys = 1.0 / complex(br["r"], br["x"])
            bc = br["b"]
            tap = br["tap"] if br["tap"] else 1.0
            tap = tap * np.exp(1j * np.deg2rad(br["shift"]))
            Ytt[k] = ys + 1j * bc / 2
            Yff[k] = Ytt[k] / (tap * np.conj(tap))
            Yft[k] = -ys / np.conj(tap)
            Ytf[k] = -ys / tap
            f[k], t[k] = self.idx[br["from"]], self.idx[br["to"]]
        self.f, self.t = f, t
        # sparse so that grids with thousands of buses fit in memory
        rows = np.arange(nl)
        self.Yf = sp.csr_matrix((np.r_[Yff, Yft], (np.r_[rows, rows], np.r_[f, t])), shape=(nl, n))
        self.Yt = sp.csr_matrix((np.r_[Ytf, Ytt], (np.r_[rows, rows], np.r_[f, t])), shape=(nl, n))
        ysh = np.array([complex(b["gs"], b["bs"]) / self.base for b in self.buses])
        Cf = sp.csr_matrix((np.ones(nl), (rows, f)), shape=(nl, n))
        Ct = sp.csr_matrix((np.ones(nl), (rows, t)), shape=(nl, n))
        self.Ybus = (Cf.T @ self.Yf + Ct.T @ self.Yt + sp.diags(ysh)).tocsr()

    def adjacency(self):
        """Bus-to-bus connectivity matrix including self connections (for PMU placement)."""
        n = self.n
        A = sp.coo_matrix((np.ones(2 * len(self.f) + n), (np.r_[self.f, self.t, np.arange(n)],
                                                           np.r_[self.t, self.f, np.arange(n)])), shape=(n, n))
        A = A.tocsr()
        A.data[:] = 1
        return A

    def q_limits_at(self, V, qd=None, atol=1e-3):
        """Infer PV bus limit contacts from voltages and actual loads (MVAr).

        Used for GridPACK diagnostics, whose voltage protocol has no switch flags.
        Reference buses and offline generators are excluded. Unlike last_qlim,
        this reports numerical contact, not the solver's PV-to-PQ decisions.
        """
        qd = np.array([b["qd"] for b in self.buses]) if qd is None else np.asarray(qd)
        qg = (V * np.conj(self.Ybus @ V)).imag * self.base + qd
        limits = {}
        for g in self.gens:
            if g["status"] and self.buses[self.idx[g["bus"]]]["type"] == 2:
                low, high = limits.get(g["bus"], (0., 0.))
                limits[g["bus"]] = low + g["qmin"], high + g["qmax"]
        contacts = {}
        for bus, (low, high) in limits.items():
            q = qg[self.idx[bus]]
            if abs(q - high) <= atol:
                contacts[bus] = "max"
            elif abs(q - low) <= atol:
                contacts[bus] = "min"
        return contacts

    def solve_pf(self, pd=None, qd=None, vset=None, tol=1e-9, max_it=30, V_init=None, *,
                 qlim=True, max_qlim_it=30):
        """Newton-Raphson power flow (reference solver; loads in MW/MVAr, vset {bus_id: pu}).
        Starts from V_init, else from the voltages stored with the case (vm/va, as MATPOWER
        does), else flat.

        With qlim=True (default), aggregate in-service generator Q limits at each
        PV bus are enforced, excluding reference buses. All violating PV buses
        switch to PQ each round (MATPOWER ENFORCE_Q_LIMS=1), avoiding extra solves
        for independent violations. Switched buses stay PQ for this call only.
        last_qlim maps their bus IDs to "min"/"max"; it resets on every call.
        it counts Newton updates across all rounds. Failure to converge or finish
        within max_qlim_it power-flow rounds returns ok=False. qlim=False retains
        the unconstrained solution used by the historical reference.csv files.
        """
        self.last_qlim = {}
        n, base = self.n, self.base
        pd = np.array([b["pd"] for b in self.buses]) if pd is None else np.asarray(pd, float)
        qd = np.array([b["qd"] for b in self.buses]) if qd is None else np.asarray(qd, float)
        types = np.array([b["type"] for b in self.buses])
        pg = np.zeros(n)
        qmin, qmax = np.zeros(n), np.zeros(n)
        if V_init is not None:
            V0, A0 = np.abs(V_init).astype(float), np.angle(V_init)
        else:
            V0 = np.array([b.get("vm", 1.0) for b in self.buses])
            A0 = np.deg2rad([b.get("va", 0.0) for b in self.buses])
        for g in self.gens:
            if g["status"]:
                i = self.idx[g["bus"]]
                pg[i] += g["pg"]; V0[i] = g["vg"]
                qmin[i] += g["qmin"]; qmax[i] += g["qmax"]
        if vset:
            for b, v in vset.items():
                V0[self.idx[b]] = v
        # buses whose generators are all off become PQ
        has_gen = np.zeros(n, bool)
        for g in self.gens:
            if g["status"]:
                has_gen[self.idx[g["bus"]]] = True
        types = np.where((types == 2) & ~has_gen, 1, types)
        pv = np.where(types == 2)[0]; pq = np.where(types == 1)[0]
        Sbus = (pg - pd - 1j * qd) / base
        V = V0 * np.exp(1j * A0)
        total_it = 0
        for _ in range(max_qlim_it if qlim else 1):
            V, it, ok = self._newton_pf(Sbus, V, pv, pq, tol, max_it)
            total_it += it
            if not ok or not qlim:
                return V, total_it, ok
            qg = (V * np.conj(self.Ybus @ V)).imag * base + qd
            # MVAr tolerance consistent with the Newton mismatch tolerance.
            eps = max(tol * base, 1e-7)
            low, high = pv[qg[pv] < qmin[pv] - eps], pv[qg[pv] > qmax[pv] + eps]
            if not len(low) and not len(high):
                return V, total_it, True
            for buses, limits, side in ((low, qmin, "min"), (high, qmax, "max")):
                for i in buses:
                    self.last_qlim[self.bus_ids[i]] = side
                    Sbus[i] = complex(Sbus[i].real, (limits[i] - qd[i]) / base)
                    types[i] = 1
            pv, pq = np.where(types == 2)[0], np.where(types == 1)[0]
        return V, total_it, False

    def _newton_pf(self, Sbus, V, pv, pq, tol, max_it):
        pvpq = np.r_[pv, pq]
        for it in range(max_it):
            mis = V * np.conj(self.Ybus @ V) - Sbus
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if not F.size or np.max(np.abs(F)) < tol:
                return V, it, True
            Ibus = self.Ybus @ V
            # MATPOWER's dSbus_dV with sparse matrices
            dV, dI, dVn = sp.diags(V), sp.diags(Ibus), sp.diags(V / np.abs(V))
            dS_dVm = dV @ (self.Ybus @ dVn).conj() + dI.conj() @ dVn
            dS_dVa = 1j * dV @ (dI - self.Ybus @ dV).conj()
            dS_dVa, dS_dVm = dS_dVa.tocsr(), dS_dVm.tocsr()
            J = sp.bmat([[dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real],
                         [dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]], format="csc")
            dx = spsolve(J, -F)
            Va = np.angle(V); Vm = np.abs(V)
            Va[pvpq] += dx[:len(pvpq)]; Vm[pq] += dx[len(pvpq):]
            V = Vm * np.exp(1j * Va)
        return V, max_it, False
