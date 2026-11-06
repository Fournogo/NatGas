"""Sticky 3-state Gaussian HMM for ramp regimes (numpy only; hmmlearn has no Python 3.14 build).

States: "Ramping down" / "Flat" / "Ramping up", fitted per fuel on d_mw_15min.
- Transitions are fixed by `p_stay` (probability of staying in the same state each 15 min;
  the rest is split evenly). Higher p_stay = fewer, longer regimes = less flicker.
- Emission means/variances and initial probabilities are fitted by EM (Baum-Welch with
  the transition matrix held fixed), initialized at the 10/50/90th percentiles.
- States are decoded with Viterbi and labeled by their fitted mean (lowest = down).
- Sequences break at gaps and at excluded (night) steps; segments share parameters.\n- Generic: `decode` also works on other series (e.g. hourly solar forecast error).

Viterbi uses the whole selected range (future steps included), so this is a look-back
labeling for visualizing, not a real-time signal.
"""

import numpy as np
import pandas as pd

LABELS = ("Ramping down", "Flat", "Ramping up")
K = 3


def _transition(p_stay: float) -> np.ndarray:
    A = np.full((K, K), (1 - p_stay) / (K - 1))
    np.fill_diagonal(A, p_stay)
    return A


def _emissions(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    B = np.exp(-((x[:, None] - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    return np.maximum(B, 1e-300)


def _forward_backward(B: np.ndarray, pi: np.ndarray, A: np.ndarray):
    T = len(B)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, np.log(c).sum()


def _viterbi(logB: np.ndarray, logpi: np.ndarray, logA: np.ndarray) -> np.ndarray:
    T = len(logB)
    psi = np.empty((T, K), dtype=np.int8)
    delta = logpi + logB[0]
    for t in range(1, T):
        cand = delta[:, None] + logA
        psi[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logB[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def fit_decode(segments: list[np.ndarray], p_stay: float, n_iter: int = 50, tol: float = 1e-4):
    """Fit on a list of 1-D sequences; return (state paths, means, stds, iterations)."""
    x_all = np.concatenate(segments)
    A = _transition(p_stay)
    mu = np.quantile(x_all, [0.1, 0.5, 0.9]).astype(float)
    var = np.full(K, x_all.var() / 2 + 1.0)
    var_floor = 1e-3 * x_all.var() + 1.0
    pi = np.full(K, 1 / K)
    prev = -np.inf
    it = 0
    for it in range(1, n_iter + 1):
        num_mu = np.zeros(K)
        num_var = np.zeros(K)
        den = np.zeros(K)
        pi_acc = np.zeros(K)
        ll = 0.0
        gammas = []
        for x in segments:
            gamma, l = _forward_backward(_emissions(x, mu, var), pi, A)
            gammas.append((x, gamma))
            ll += l
            pi_acc += gamma[0]
            den += gamma.sum(axis=0)
            num_mu += gamma.T @ x
        mu = num_mu / np.maximum(den, 1e-12)
        for x, gamma in gammas:
            num_var += (gamma * (x[:, None] - mu) ** 2).sum(axis=0)
        var = np.maximum(num_var / np.maximum(den, 1e-12), var_floor)
        pi = np.maximum(pi_acc / pi_acc.sum(), 1e-6)
        pi /= pi.sum()
        if abs(ll - prev) <= tol * abs(ll):
            break
        prev = ll

    order = np.argsort(mu)  # state index -> rank (0 = lowest mean)
    rank = np.empty(K, dtype=int)
    rank[order] = np.arange(K)
    logA, logpi = np.log(A), np.log(pi)
    paths = [rank[_viterbi(np.log(_emissions(x, mu, var)), logpi, logA)] for x in segments]
    return paths, mu[order], np.sqrt(var[order]), it


def _segments(g: pd.DataFrame, value: str, t: str, step: pd.Timedelta) -> list[np.ndarray]:
    """Index arrays of runs of consecutive usable rows (break at gaps / excluded rows)."""
    ok = (g[value].notna() & ~g["excluded"]).to_numpy()
    contiguous = (g[t].diff() == step).to_numpy()
    runs, cur = [], []
    for i in range(len(g)):
        if ok[i] and cur and contiguous[i] and ok[i - 1]:
            cur.append(i)
        else:
            if cur:
                runs.append(np.array(cur))
            cur = [i] if ok[i] else []
    if cur:
        runs.append(np.array(cur))
    return runs


def decode(d: pd.DataFrame, p_stay: float, value: str = "d_mw_15min", t: str = "interval_end_utc",
           group: str = "fuel", step: pd.Timedelta = pd.Timedelta(minutes=15),
           labels: tuple = LABELS) -> tuple[pd.Series, pd.DataFrame]:
    """Regime per row of `d` (NaN where not usable) and fitted parameters per `group`.
    `d` needs columns value, t, group and a boolean `excluded`; `labels` name the states
    from lowest to highest fitted mean."""
    states = pd.Series(pd.NA, index=d.index, dtype="object")
    params = []
    for key, g in d.groupby(group, sort=False):
        g = g.sort_values(t)
        runs = _segments(g, value, t, step)
        if not runs:
            continue
        x = g[value].to_numpy(dtype=float)
        paths, mu, sd, it = fit_decode([x[r] for r in runs], p_stay)
        for r, p in zip(runs, paths):
            states.loc[g.index[r]] = np.array(labels)[p]
        params.append({group: key, **{f"mean {l}": m for l, m in zip(labels, mu)},
                       **{f"std {l}": s for l, s in zip(labels, sd)}, "EM iterations": it})
    return states, pd.DataFrame(params).set_index(group)
