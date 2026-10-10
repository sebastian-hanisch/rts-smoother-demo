"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (Median über SWEEP_SEEDS).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import numpy as np
import pytest

import rts_constants as C
import rts_evaluation as ev


@pytest.fixture(scope="module")
def sigma_rows():
    return {r["sigma"]: r for r in ev.sweep_sigma(ev.Settings())}


@pytest.fixture(scope="module")
def lag_rows():
    return ev.sweep_lag(ev.Settings())


@pytest.fixture(scope="module")
def tunnel_rows():
    return {r["tunnel"]: r for r in ev.sweep_tunnel(ev.Settings())}


def test_standard_track_numbers():
    a = ev.analyse(ev.Settings())
    assert a.rmse_f == pytest.approx(1.07, abs=0.1) and a.rmse_s == pytest.approx(0.85, abs=0.1)
    assert 1 - a.rmse_s / a.rmse_f == pytest.approx(0.20, abs=0.07)
    assert a.rmse_lag < a.rmse_f_w


def test_smoother_beats_filter_at_every_noise_level_and_gain_grows_with_noise(sigma_rows):
    gains = [sigma_rows[s]["gain"] for s in C.SIGMA_SWEEP]
    assert all(g > 0.2 for g in gains)
    assert gains[0] == pytest.approx(0.29, abs=0.06) and gains[2] == pytest.approx(0.31, abs=0.06) and gains[-1] == pytest.approx(0.46, abs=0.07)
    assert gains[-1] > gains[0]


def test_lag_needs_long_delay_for_most_of_the_gain(lag_rows):
    share = {L: ev.lag_share(lag_rows, L) for L in C.LAG_SWEEP}
    assert share[0] == pytest.approx(0.0, abs=1e-9)
    assert share[5] == pytest.approx(0.21, abs=0.12)
    assert share[20] == pytest.approx(0.71, abs=0.15)
    assert share[40] == pytest.approx(0.85, abs=0.15)
    assert share[5] < share[20] < share[40]


def test_lag_curve_is_monotone_down_to_the_full_smoother(lag_rows):
    r = [x["rmse"] for x in lag_rows]
    assert all(b <= a + 0.03 for a, b in zip(r, r[1:]))   # Rauschen der 40 Fahrten
    assert lag_rows[-1]["rmse"] >= lag_rows[-1]["smooth"] - 1e-9


def test_tunnel_without_sensors_the_smoother_bridges_the_gap(tunnel_rows):
    t = tunnel_rows[40]
    assert t["f_aus"] == pytest.approx(28, abs=5) and t["s_aus"] == pytest.approx(7.0, abs=2)
    assert t["f_aus"] > 3 * t["s_aus"]
    assert t["f_an"] == pytest.approx(1.27, abs=0.3) and t["s_an"] == pytest.approx(1.04, abs=0.3)
    assert t["s_an"] < t["f_an"]
