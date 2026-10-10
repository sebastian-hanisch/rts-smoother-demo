"""Orakel-Test rts-smoother-demo: der Glätter gegen ANDERE Rechenwege.

1. Alle geglätteten Zustände UND ihre Kovarianzen gegen die Lösung der ganzen Fahrt als gewichtetes Kleinste-Quadrate-System,
2. der Verzögerungs-Glätter gegen den vollen Glätter auf abgeschnittenen Daten,
3. Konsistenz: mittlere NEES des Glätters über viele Fahrten gleich der Zustandsdimension,
4. die angezeigten Kennzahlen gegen eine unabhängige Neuberechnung.
"""

import numpy as np
import pytest

import rts_algorithm as A
import rts_constants as C
import rts_evaluation as ev
import rts_scenario as S


def _batch(z, w, q, sigma_z, sigma_w, use_wheel, v_mean, v_std):
    """Ganze Fahrt als gewichtetes Kleinste-Quadrate-System über x_0..x_{T-1}: Zustände und marginale Kovarianzen."""
    T = len(z)
    F, Q = A.cv_model(1.0, q)
    Lq = np.linalg.cholesky(np.linalg.inv(Q))
    rows, rhs = [], []

    def add(blocks, b, weight):
        row = np.zeros(2 * T)
        for t, blk in blocks:
            row[2 * t:2 * t + 2] += blk
        rows.append(weight * row)
        rhs.append(weight * b)

    add([(0, np.array([1.0, 0.0]))], z[0], 1 / sigma_z)
    add([(0, np.array([0.0, 1.0]))], v_mean, 1 / v_std)
    for t in range(T - 1):
        for j in range(2):
            wj = Lq[:, j]
            add([(t + 1, wj), (t, -(F.T @ wj))], 0.0, 1.0)
    for t in range(1, T):
        if not np.isnan(z[t]):
            add([(t, A.H_GPS)], z[t], 1 / sigma_z)
    if use_wheel:
        for t in range(T):
            add([(t, A.H_WHEEL)], w[t], 1 / sigma_w)
    M, b = np.array(rows), np.array(rhs)
    x = np.linalg.lstsq(M, b, rcond=None)[0].reshape(T, 2)
    cov = np.linalg.inv(M.T @ M)
    blocks = np.array([cov[2 * t:2 * t + 2, 2 * t:2 * t + 2] for t in range(T)])
    return x, blocks


def test_smoother_equals_batch_least_squares_at_every_time_step():
    for seed in range(5):
        for wheel in (False, True):
            tr = S.simulate(seed, 10.0, 0.3, 8)
            z, w = tr.z[:40].copy(), tr.w[:40]
            z[15:25] = np.nan
            run = A.run_filter(z, w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            F, _ = A.cv_model(1.0, 0.3)
            xs, Ps = A.rts_smooth(run, F)
            x_b, P_b = _batch(z, w, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            assert xs == pytest.approx(x_b, rel=1e-6, abs=1e-6), (seed, wheel)
            assert Ps == pytest.approx(P_b, rel=1e-6, abs=1e-9), (seed, wheel)


def test_fixed_lag_equals_full_smoother_on_truncated_data():
    tr = S.simulate(9, 10.0, 0.3, 10)
    z, w = tr.z[:50].copy(), tr.w[:50]
    F, _ = A.cv_model(1.0, 0.3)
    run = A.run_filter(z, w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
    for lag in (1, 3, 7, 15):
        x_l, P_l = A.fixed_lag_smooth(run, F, lag)
        for k in (0, 5, 20, 49 - lag):
            run_cut = A.run_filter(z[:k + lag + 1], w[:k + lag + 1], 1.0, 0.3, 10.0, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
            xs, Ps = A.rts_smooth(run_cut, F)
            assert x_l[k] == pytest.approx(xs[k], rel=1e-9, abs=1e-9), (lag, k)
            assert P_l[k] == pytest.approx(Ps[k], rel=1e-9, abs=1e-12), (lag, k)


def test_mean_nees_of_the_smoother_equals_state_dimension_over_many_tracks():
    for wheel in (False, True):
        vals = []
        for seed in range(150):
            tr = S.simulate(500 + seed, 10.0, 0.3, 20)
            run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            F, _ = A.cv_model(1.0, 0.3)
            xs, Ps = A.rts_smooth(run, F)
            e = np.c_[xs[:, 0] - tr.s, xs[:, 1] - tr.v]
            vals.append(np.mean([A.nees(e[k], Ps[k]) for k in range(C.N_BURN, C.T_STEPS)]))
        assert np.mean(vals) == pytest.approx(2.0, abs=0.3), wheel


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11, 40):
        s = ev.Settings(seed=seed, tunnel=20, lag=7)
        a = ev.analyse(s)
        tr = S.simulate(seed, s.sigma_z, s.q, s.tunnel)
        F, _ = A.cv_model(1.0, s.q)
        run = A.run_filter(tr.z, tr.w, 1.0, s.q, s.sigma_z, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
        x_b, P_b = _batch(tr.z, tr.w, s.q, s.sigma_z, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
        sl = slice(C.N_BURN, None)
        assert a.rmse_f == pytest.approx(np.sqrt(np.mean((run.x[sl, 0] - tr.s[sl]) ** 2)))
        assert a.rmse_s == pytest.approx(np.sqrt(np.mean((x_b[sl, 0] - tr.s[sl]) ** 2)), rel=1e-6)
        tun = tr.tunnel.copy()
        tun[:C.N_BURN] = False
        assert a.rmse_s_tunnel == pytest.approx(np.sqrt(np.mean((x_b[tun, 0] - tr.s[tun]) ** 2)), rel=1e-6)
        n = C.T_STEPS - s.lag
        xl_ref = np.array([x_b[k] for k in range(n)])  # Referenz nur für die Dimension; Lag-Wert gegen abgeschnittene Daten unten
        assert a.x_lag.shape == xl_ref.shape
        assert a.sd_s[50] == pytest.approx(np.sqrt(P_b[50, 0, 0]), rel=1e-6)


def test_lag_window_metrics_use_the_same_time_points():
    a = ev.analyse(ev.Settings(lag=10, seed=4))
    n = len(a.x_lag)
    m = np.arange(n) >= C.N_BURN
    assert a.rmse_f_w == pytest.approx(np.sqrt(np.mean(a.err_f[:n][m] ** 2)))
    assert a.rmse_s_w == pytest.approx(np.sqrt(np.mean(a.err_s[:n][m] ** 2)))
    assert a.rmse_lag == pytest.approx(np.sqrt(np.mean((a.x_lag[m, 0] - a.track.s[:n][m]) ** 2)))
