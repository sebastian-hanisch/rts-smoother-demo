"""Kalman-Filter und Rauch-Tung-Striebel-Glätter, numpy-only. Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben.

Modell: Zustand x = (Position s, Geschwindigkeit v), konstante Geschwindigkeit mit zufälliger Beschleunigung (Rauschen q in m/s^2).
Messungen: GPS misst s, Radsensor misst v. Der Filter ist derselbe wie im Stück „Kalman-Filter“; neu ist die Rückwärtsrechnung.
"""

from dataclasses import dataclass

import numpy as np

H_GPS = np.array([1.0, 0.0])
H_WHEEL = np.array([0.0, 1.0])


def cv_model(dt, q):
    """Übergangsmatrix F und Prozessrauschen Q des Modells mit konstanter Geschwindigkeit und weißem Beschleunigungsrauschen q."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q ** 2 * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return F, Q


def predict(x, P, F, Q):
    """Vorhersage: Zustand mit dem Modell fortschreiben, Unsicherheit wächst um Q."""
    return F @ x, F @ P @ F.T + Q


def update_scalar(x, P, z, h, r):
    """Korrektur mit einer skalaren Messung z = h·x + Rauschen (Varianz r), Joseph-Form. Gibt (x, P, K, nu, S) zurück."""
    S = float(h @ P @ h + r)
    K = (P @ h) / S
    nu = float(z - h @ x)
    x_new = x + K * nu
    I_KH = np.eye(len(x)) - np.outer(K, h)
    P_new = I_KH @ P @ I_KH.T + np.outer(K, K) * r
    return x_new, P_new, K, nu, S


def nees(err, P):
    """Normiertes Fehlerquadrat err' P^-1 err; Erwartung bei ehrlicher Schätzung: Zustandsdimension."""
    return float(err @ np.linalg.solve(P, err))


@dataclass(frozen=True)
class FilterRun:
    x: np.ndarray        # (T, 2) gefilterter Zustand nach der Korrektur (nur Daten bis t)
    P: np.ndarray        # (T, 2, 2) Kovarianz
    x_pred: np.ndarray   # (T, 2) Vorhersage vor der Korrektur (für den Glätter nötig)
    P_pred: np.ndarray   # (T, 2, 2)


def run_filter(z, w, dt, q, sigma_z, sigma_w, use_wheel, v_prior_mean, v_prior_std):
    """Kalman-Filter über eine ganze Spur; z: GPS-Positionen (NaN = Ausfall), w: Radsensor-Geschwindigkeiten."""
    T = len(z)
    F, Q = cv_model(dt, q)
    r_gps, r_wheel = sigma_z ** 2, sigma_w ** 2
    x = np.array([z[0], v_prior_mean], float)
    P = np.diag([r_gps, v_prior_std ** 2])
    xs, Ps = np.zeros((T, 2)), np.zeros((T, 2, 2))
    xp, Pp = np.zeros((T, 2)), np.zeros((T, 2, 2))
    for t in range(T):
        if t > 0:
            x, P = predict(x, P, F, Q)
            xp[t], Pp[t] = x, P
            if not np.isnan(z[t]):
                x, P, _, _, _ = update_scalar(x, P, z[t], H_GPS, r_gps)
        else:
            xp[t], Pp[t] = x, P
        if use_wheel:
            x, P, _, _, _ = update_scalar(x, P, w[t], H_WHEEL, r_wheel)
        xs[t], Ps[t] = x, P
    return FilterRun(xs, Ps, xp, Pp)


def smoother_gains(F, P_filt, P_pred):
    """Glätter-Verstärkung C_k = P_k|k F^T (P_k+1|k)^-1 für k = 0 .. T-2: wie stark die Zukunft auf Zeitpunkt k zurückwirkt."""
    return np.array([P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1]) for k in range(len(P_filt) - 1)])


def rts_smooth(run, F):
    """Rauch-Tung-Striebel-Glätter über die ganze Spur: Schätzung jedes Zeitpunkts aus ALLEN Messungen.

    Rückwärts von T-1 nach 0:  x_k|T = x_k|k + C_k (x_k+1|T - x_k+1|k),  P_k|T = P_k|k + C_k (P_k+1|T - P_k+1|k) C_k^T.
    """
    T = len(run.x)
    C = smoother_gains(F, run.P, run.P_pred)
    xs, Ps = run.x.copy(), run.P.copy()
    for k in range(T - 2, -1, -1):
        xs[k] = run.x[k] + C[k] @ (xs[k + 1] - run.x_pred[k + 1])
        Ps[k] = run.P[k] + C[k] @ (Ps[k + 1] - run.P_pred[k + 1]) @ C[k].T
    return xs, Ps


def fixed_lag_smooth(run, F, lag):
    """Festverzögerungs-Glätter: Schätzung des Zeitpunkts k aus den Messungen bis k + lag (Echtzeit mit `lag` Sekunden Verspätung).

    Gibt (x, P) für die Zeitpunkte k = 0 .. T-1-lag zurück. lag = 0 ist der Filter, lag = T-1 der volle Glätter.
    Alle Ziele werden gleichzeitig rückwärts gerechnet (dieselben Verstärkungen C_k, gestartet bei x_k+lag|k+lag).
    """
    T = len(run.x)
    lag = int(min(max(lag, 0), T - 1))
    ks = np.arange(T - lag)
    cur, Pcur = run.x[ks + lag].copy(), run.P[ks + lag].copy()
    if lag == 0:
        return cur, Pcur
    C = smoother_gains(F, run.P, run.P_pred)
    for s in range(lag):
        m = ks + lag - 1 - s
        cur = run.x[m] + np.einsum("tij,tj->ti", C[m], cur - run.x_pred[m + 1])
        Pcur = run.P[m] + np.einsum("tij,tjk,tlk->til", C[m], Pcur - run.P_pred[m + 1], C[m])
    return cur, Pcur
