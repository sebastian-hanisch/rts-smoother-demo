"""Einheiten des Glätter-Kerns gegen Handrechnung auf Mini-Instanzen und Randfälle."""

import numpy as np
import pytest

import rts_algorithm as A
import rts_constants as C
import rts_scenario as S


def _hand_run():
    """Zwei Zeitpunkte, F = I (nur zur Handrechnung): x_0|0 = (0,0), x_1|1 = (2,2), x_1|0 = (1,1), P_0|0 = P_1|1 = I, P_1|0 = 2 I."""
    x = np.array([[0.0, 0.0], [2.0, 2.0]])
    P = np.array([np.eye(2), np.eye(2)])
    xp = np.array([[0.0, 0.0], [1.0, 1.0]])
    Pp = np.array([np.eye(2), 2 * np.eye(2)])
    return A.FilterRun(x, P, xp, Pp)


def test_smoother_gain_hand_example():
    run = _hand_run()
    C_ = A.smoother_gains(np.eye(2), run.P, run.P_pred)
    assert C_.shape == (1, 2, 2)
    assert C_[0] == pytest.approx(0.5 * np.eye(2))


def test_rts_one_backward_step_hand_example():
    # Handrechnung: x_0|T = 0 + 0.5 (2 - 1) = 0.5, P_0|T = 1 + 0.25 (1 - 2) = 0.75
    xs, Ps = A.rts_smooth(_hand_run(), np.eye(2))
    assert xs[1].tolist() == [2.0, 2.0] and xs[0] == pytest.approx([0.5, 0.5])
    assert Ps[1] == pytest.approx(np.eye(2)) and Ps[0] == pytest.approx(0.75 * np.eye(2))


def test_last_step_of_the_smoother_is_the_filter():
    tr = S.simulate(3, 10.0, 0.3, 20)
    run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
    F, _ = A.cv_model(1.0, 0.3)
    xs, Ps = A.rts_smooth(run, F)
    assert xs[-1] == pytest.approx(run.x[-1]) and Ps[-1] == pytest.approx(run.P[-1])


def test_smoothing_never_increases_uncertainty():
    for seed in range(10):
        for wheel in (False, True):
            tr = S.simulate(seed, 10.0, 0.3, 15)
            run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, wheel, C.V_START_MEAN, C.V_PRIOR_STD)
            F, _ = A.cv_model(1.0, 0.3)
            _, Ps = A.rts_smooth(run, F)
            for k in range(len(Ps)):
                assert np.all(np.linalg.eigvalsh(run.P[k] - Ps[k]) >= -1e-9), (seed, wheel, k)
                assert np.allclose(Ps[k], Ps[k].T)


def test_fixed_lag_zero_is_the_filter_and_full_lag_is_the_smoother_at_time_zero():
    tr = S.simulate(5, 10.0, 0.3, 10)
    run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
    F, _ = A.cv_model(1.0, 0.3)
    x0, P0 = A.fixed_lag_smooth(run, F, 0)
    assert x0 == pytest.approx(run.x) and P0 == pytest.approx(run.P)
    xs, Ps = A.rts_smooth(run, F)
    xl, Pl = A.fixed_lag_smooth(run, F, C.T_STEPS - 1)
    assert xl.shape == (1, 2) and xl[0] == pytest.approx(xs[0]) and Pl[0] == pytest.approx(Ps[0])


def test_fixed_lag_clips_a_too_large_lag():
    tr = S.simulate(5, 10.0, 0.3, 10)
    run = A.run_filter(tr.z, tr.w, 1.0, 0.3, 10.0, C.SIGMA_WHEEL, True, C.V_START_MEAN, C.V_PRIOR_STD)
    F, _ = A.cv_model(1.0, 0.3)
    xl, _ = A.fixed_lag_smooth(run, F, 10_000)
    assert xl.shape == (1, 2)


def test_simulation_is_reproducible_and_regler_scale_the_same_randomness():
    a = S.simulate(7, 10.0, 0.3, 20)
    b = S.simulate(7, 10.0, 0.3, 20)
    assert np.array_equal(a.s, b.s) and np.array_equal(a.z, b.z, equal_nan=True)
    c = S.simulate(7, 20.0, 0.3, 20)
    assert np.array_equal(a.s, c.s)
    have = ~a.tunnel
    assert np.allclose((c.z - c.s)[have], 2 * (a.z - a.s)[have])


def test_tunnel_mask_and_nan_positions():
    for length in (0, 1, 20, 40):
        tr = S.simulate(1, 10.0, 0.3, length)
        assert tr.tunnel.sum() == length
        assert np.isnan(tr.z[tr.tunnel]).all() and not np.isnan(tr.z[~tr.tunnel]).any()


def test_constants_are_consistent():
    assert C.TUNNEL_START + C.TUNNEL_MAX < C.T_STEPS and C.N_BURN < C.TUNNEL_START
    assert C.LAG_MAX < C.T_STEPS and max(C.LAG_SWEEP) <= C.LAG_MAX and C.LAG_MIN == 0 and 0 in C.LAG_SWEEP
