"""Szenario: ein Lieferfahrzeug fährt eine gerade Strecke; GPS misst die Position, der Radsensor die Geschwindigkeit."""

from dataclasses import dataclass

import numpy as np

import rts_algorithm as A
import rts_constants as C


@dataclass(frozen=True)
class Track:
    t: np.ndarray        # Zeit in s
    s: np.ndarray        # wahre Position in m
    v: np.ndarray        # wahre Geschwindigkeit in m/s
    z: np.ndarray        # GPS-Messung, NaN im Tunnel
    w: np.ndarray        # Radsensor-Messung
    tunnel: np.ndarray   # bool je Schritt: GPS ausgefallen


def tunnel_mask(length):
    mask = np.zeros(C.T_STEPS, bool)
    mask[C.TUNNEL_START:C.TUNNEL_START + max(0, int(length))] = True
    return mask


def simulate(seed, sigma_z, q_true, tunnel_len):
    """Wahre Fahrt plus Messungen. Alle Zufallszahlen kommen in fester Reihenfolge aus einem Generator; die Regler skalieren sie nur.

    So ändert ein Regler die Spur stetig (derselbe Zufall, anderes Rauschen) und nicht jedes Mal komplett.
    """
    rng = np.random.default_rng(seed)
    n_proc = rng.standard_normal((C.T_STEPS, 2))
    n_gps = rng.standard_normal(C.T_STEPS)
    n_whl = rng.standard_normal(C.T_STEPS)
    n_v0 = float(rng.standard_normal())
    F, Q = A.cv_model(C.DT, q_true)
    L = np.linalg.cholesky(Q)
    s, v = np.zeros(C.T_STEPS), np.zeros(C.T_STEPS)
    x = np.array([0.0, C.V_START_MEAN + C.V_START_STD * n_v0])
    for k in range(C.T_STEPS):
        s[k], v[k] = x
        x = F @ x + L @ n_proc[k]
    mask = tunnel_mask(tunnel_len)
    z = s + sigma_z * n_gps
    z[mask] = np.nan
    w = v + C.SIGMA_WHEEL * n_whl
    return Track(np.arange(C.T_STEPS) * C.DT, s, v, z, w, mask)
