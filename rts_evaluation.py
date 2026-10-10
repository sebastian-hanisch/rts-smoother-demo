"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass, replace

import numpy as np

import rts_algorithm as A
import rts_constants as C
import rts_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma_z: float = C.DEFAULT_SIGMA_Z
    q: float = C.DEFAULT_Q
    tunnel: int = C.DEFAULT_TUNNEL
    wheel: str = C.DEFAULT_WHEEL
    lag: int = C.DEFAULT_LAG
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    track: S.Track
    run: A.FilterRun
    xs: np.ndarray             # geglätteter Zustand (T, 2)
    Ps: np.ndarray
    x_lag: np.ndarray          # Festverzögerungs-Glätter, gültig für k = 0 .. T-1-lag
    P_lag: np.ndarray
    err_f: np.ndarray          # Positionsfehler Filter / Glätter (T,)
    err_s: np.ndarray
    sd_f: np.ndarray
    sd_s: np.ndarray
    rmse_f: float              # Positionsfehler ab N_BURN über ALLE Schritte (auch im Tunnel)
    rmse_s: float
    rmse_lag: float            # Lag-Glätter auf den Zeitpunkten k <= T-1-lag
    rmse_f_w: float            # Filter und Glätter auf genau demselben Fenster wie rmse_lag
    rmse_s_w: float
    rmse_f_tunnel: float       # nur Tunnelschritte (NaN ohne Tunnel)
    rmse_s_tunnel: float
    nees_f: float
    nees_s: float
    coverage_s: float


def run_settings(settings):
    track = S.simulate(settings.seed, settings.sigma_z, settings.q, settings.tunnel)
    run = A.run_filter(track.z, track.w, C.DT, settings.q, settings.sigma_z, C.SIGMA_WHEEL, settings.wheel == "an", C.V_START_MEAN, C.V_PRIOR_STD)
    return track, run


def _rmse(err, mask):
    return float(np.sqrt(np.mean(err[mask] ** 2))) if np.any(mask) else float("nan")


def _mean_nees(x, P, truth, mask):
    e = np.c_[x[:, 0] - truth.s[: len(x)], x[:, 1] - truth.v[: len(x)]]
    vals = [A.nees(e[k], P[k]) for k in range(len(x)) if mask[k]]
    return float(np.mean(vals)) if vals else float("nan")


def analyse(settings):
    track, run = run_settings(settings)
    F, _ = A.cv_model(C.DT, settings.q)
    xs, Ps = A.rts_smooth(run, F)
    x_lag, P_lag = A.fixed_lag_smooth(run, F, settings.lag)
    T = C.T_STEPS
    err_f = run.x[:, 0] - track.s
    err_s = xs[:, 0] - track.s
    idx = np.arange(T)
    after = idx >= C.N_BURN
    n_lag = len(x_lag)
    err_l = x_lag[:, 0] - track.s[:n_lag]
    mask_l = np.arange(n_lag) >= C.N_BURN
    sd_f, sd_s = np.sqrt(run.P[:, 0, 0]), np.sqrt(Ps[:, 0, 0])
    inside = np.abs(err_s[after]) <= 1.96 * sd_s[after]
    return Analysis(
        settings, track, run, xs, Ps, x_lag, P_lag, err_f, err_s, sd_f, sd_s,
        _rmse(err_f, after), _rmse(err_s, after), _rmse(err_l, mask_l),
        _rmse(err_f[:n_lag], mask_l), _rmse(err_s[:n_lag], mask_l),
        _rmse(err_f, after & track.tunnel), _rmse(err_s, after & track.tunnel),
        _mean_nees(run.x, run.P, track, after), _mean_nees(xs, Ps, track, after), float(np.mean(inside)),
    )


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _over_seeds(settings, fn):
    return [fn(analyse(replace(settings, seed=s))) for s in C.SWEEP_SEEDS]


def sweep_sigma(settings):
    """GPS-Rauschen variieren: Positionsfehler von Filter und Glätter (Median über die festen Fahrten)."""
    rows = []
    for sg in C.SIGMA_SWEEP:
        res = _over_seeds(replace(settings, sigma_z=sg), lambda a: (a.rmse_f, a.rmse_s))
        f = [r[0] for r in res]
        s = [r[1] for r in res]
        rows.append({"sigma": sg, "filter": float(np.median(f)), "smooth": float(np.median(s)),
                     "gain": float(np.median([1.0 - b / a for a, b in zip(f, s)]))})
    return rows


def sweep_lag(settings):
    """Verzögerung variieren: Positionsfehler des Festverzögerungs-Glätters gegen Filter und vollen Glätter.

    Je Lag wird auf denselben Zeitpunkten k = N_BURN .. T-1-lag verglichen (der Lag-Glätter kennt die letzten `lag` Zeitpunkte noch nicht).
    """
    F, _ = A.cv_model(C.DT, settings.q)
    acc = {L: {"lag": [], "filter": [], "smooth": []} for L in C.LAG_SWEEP}
    for s in C.SWEEP_SEEDS:
        track, run = run_settings(replace(settings, seed=s))
        xs, _ = A.rts_smooth(run, F)
        for L in C.LAG_SWEEP:
            x_l, _ = A.fixed_lag_smooth(run, F, L)
            n = len(x_l)
            m = np.arange(n) >= C.N_BURN
            acc[L]["lag"].append(_rmse(x_l[:, 0] - track.s[:n], m))
            acc[L]["filter"].append(_rmse(run.x[:n, 0] - track.s[:n], m))
            acc[L]["smooth"].append(_rmse(xs[:n, 0] - track.s[:n], m))
    return [{"lag": L, "rmse": float(np.median(acc[L]["lag"])), "filter": float(np.median(acc[L]["filter"])),
             "smooth": float(np.median(acc[L]["smooth"]))} for L in C.LAG_SWEEP]


def sweep_tunnel(settings):
    """Tunnellänge variieren: Positionsfehler im Tunnel, Filter gegen Glätter, mit und ohne Radsensor."""
    rows = []
    for L in (10, 20, 30, 40):
        row = {"tunnel": L}
        for wheel in ("an", "aus"):
            res = _over_seeds(replace(settings, tunnel=L, wheel=wheel), lambda a: (a.rmse_f_tunnel, a.rmse_s_tunnel))
            row[f"f_{wheel}"] = float(np.median([r[0] for r in res]))
            row[f"s_{wheel}"] = float(np.median([r[1] for r in res]))
        rows.append(row)
    return rows


def lag_share(rows, lag):
    """Anteil des Gewinns (Filter -> voller Glätter), den ein gegebener Lag erreicht; None, wenn es keinen Gewinn gibt."""
    r = next(x for x in rows if x["lag"] == lag)
    gap = r["filter"] - r["smooth"]
    if gap <= 1e-9:
        return None
    return float((r["filter"] - r["rmse"]) / gap)
