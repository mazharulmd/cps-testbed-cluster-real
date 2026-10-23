"""Linear PMU state estimation with chi-square bad data detection.

With PMU phasors the measurement model is linear, z = H V + e, so the weighted
least-squares estimate is V = (H^H W H)^-1 H^H W z. The residual objective
J = sum |r_k|^2 / sigma_k^2 follows a chi-square distribution with
2m - 2n degrees of freedom when the data are clean; J above the threshold flags
bad (possibly falsified) data. The largest normalized residual points to the
suspicious channel.
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import chi2


def sigma_from_polar_noise(mag, ang):
    """Relative rectangular per-component std-dev for independent polar noise.

    To first order, averaging the real/imaginary variances gives
    |z|**2 * (mag**2 + ang**2) / 2. Equal magnitude and angle noise is isotropic;
    for unequal values this is a scalar approximation to the covariance.
    The estimator retains its measured-|z| weighting and absolute noise floor.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (mag, ang)):
        raise ValueError("polar noise standard deviations must be finite and nonnegative")
    return math.hypot(mag, ang) / math.sqrt(2)


class StateEstimator:
    def __init__(self, layout, sigma=0.002, alpha=0.01, pseudo_sigma=0.05):
        self.L = layout
        self.sigma = sigma              # relative std-dev of a PMU phasor (rectangular)
        self.alpha = alpha              # false-alarm probability of the chi-square test
        self.pseudo_sigma = pseudo_sigma
        self.last = np.ones(layout.net.n, complex)

    def estimate(self, meas, exclude=()):
        """meas: {(pmu_idx, ch_idx): complex}. Returns dict with estimate and test results."""
        net, n = self.L.net, self.L.net.n
        keys = [k for k in meas if k not in exclude]
        rows = [self.L.row[k] for k in keys]
        H = self.L.H[rows] if rows else sp.csr_matrix((0, n), dtype=complex)
        z = np.array([meas[k] for k in keys], complex)
        sig = np.maximum(self.sigma * np.abs(z), 1e-4)
        W = 1.0 / sig ** 2
        Hh = H.conj().T.tocsr()
        G = (Hh @ sp.diags(W) @ H).tocsc()
        rhs = Hh @ (W * z)
        # PMU observability is topological: every bus must be seen by a received PMU
        seen = set()
        for p in {k[0] for k in keys}:
            seen |= self.L.sees[p]
        observable = len(rows) >= n and len(seen) == n
        V, lu = None, None
        if observable:
            try:
                # G is Hermitian positive definite: symmetric ordering, diagonal pivots
                lu = splu(G, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                          options={"SymmetricMode": True})
                V = lu.solve(rhs)
                observable = bool(np.all(np.isfinite(V)))
            except RuntimeError:
                observable = False
        pseudo = 0
        if not observable:
            # keep the estimator solvable with weak pseudo-measurements from the last estimate
            Gp = (G + sp.identity(n, format="csc") / self.pseudo_sigma ** 2).tocsc()
            V = splu(Gp).solve(rhs + self.last / self.pseudo_sigma ** 2)
            pseudo = n
        r = z - H @ V
        J = float(np.sum(np.abs(r) ** 2 * W))
        dof = max(2 * len(rows) - 2 * n, 1) if observable else max(2 * len(rows), 1)
        thr = float(chi2.ppf(1 - self.alpha, dof))
        alarm = bool(observable and J > thr)
        # normalized residuals r_k / sqrt(Omega_kk), Omega = R - H G^-1 H^H; only needed to
        # pick the suspicious channel after an alarm
        worst = None
        if alarm and len(rows) > n:
            worst = self._largest_normalized_residual(keys, rows, H, Hh, lu, r, sig)
        self.last = V
        return {"V": V, "J": J, "threshold": thr, "dof": dof, "alarm": alarm,
                "observable": bool(observable), "m": len(rows), "worst": worst}

    EXACT_ROWS = 1500      # up to this many channels every normalized residual is computed
    CANDIDATE_PMUS = 25    # beyond it, only the channels of the PMUs with the largest raw residuals

    def _largest_normalized_residual(self, keys, rows, H, Hh, lu, r, sig):
        """Channel with the largest normalized residual r_k / sqrt(Omega_kk),
        Omega_kk = sigma_k^2 - h_k G^-1 h_k^H. On large grids the exact value is computed
        only for the channels of the PMUs with the largest raw residuals |r_k| / sigma_k:
        a falsified channel stands out there, and a critical channel (Omega_kk ~ 0) has a
        residual of ~0 anyway."""
        cand = np.arange(len(rows))
        if len(rows) > self.EXACT_ROWS:
            raw = np.abs(r) / sig
            score = {}
            for i, k in enumerate(keys):
                score[k[0]] = max(score.get(k[0], 0.0), raw[i])
            top = set(sorted(score, key=score.get, reverse=True)[:self.CANDIDATE_PMUS])
            cand = np.array([i for i, k in enumerate(keys) if k[0] in top])
        # one right-hand side at a time: SuperLU's multi-RHS solve is much slower here
        Hhc = Hh.tocsc()
        diag = np.array([np.real(H[i] @ lu.solve(Hhc[:, i].toarray().ravel()))[0] for i in cand])
        omega = sig[cand] ** 2 - diag
        rn = np.abs(r[cand]) / np.sqrt(np.maximum(omega, 1e-12))
        # channels can tie exactly (critical pairs); take the first so the choice is deterministic
        j = int(np.flatnonzero(rn >= rn.max() * (1 - 1e-9))[0])
        return keys[cand[j]], float(rn[j])

    def estimate_with_bdd(self, meas, max_removals=3):
        """Run the chi-square test; on an alarm remove the PMU with the largest normalized
        residual and re-estimate, up to max_removals PMUs."""
        removed, excl = [], set()
        res = self.estimate(meas)
        first = res
        while res["alarm"] and res["worst"] and len(removed) < max_removals:
            p = res["worst"][0][0]
            removed.append(p)
            excl |= {k for k in meas if k[0] == p}
            res = self.estimate(meas, exclude=excl)
        res["removed_pmus"] = removed
        res["first_J"], res["first_alarm"] = first["J"], first["alarm"]
        res["first_threshold"], res["first_observable"] = first["threshold"], first["observable"]
        return res
