"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import rts_constants as C
import rts_evaluation as ev
import rts_presets as P


def _settings_from(p):
    return ev.Settings(sigma_z=p["sz"], q=p["q"], tunnel=p["tun"], wheel=p["rad"], lag=p["lag"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_Z_MIN <= p["sz"] <= C.SIGMA_Z_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX, name
        assert C.TUNNEL_MIN <= p["tun"] <= C.TUNNEL_MAX and p["rad"] in C.WHEEL_CHOICES, name
        assert C.LAG_MIN <= p["lag"] <= C.LAG_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standardfahrt (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    lag = P.SETTING_SPECS["lag_slider"]
    assert P.clamp_value(lag, 1000) == C.LAG_MAX and P.clamp_value(lag, -3) == C.LAG_MIN
    rad = P.SETTING_SPECS["wheel_select"]
    assert P.clamp_value(rad, "an") == "an" and P.clamp_value(rad, "kaputt") is None


def test_the_presets_show_what_their_help_text_claims():
    base = C.PRESETS["Standardfahrt (Voreinstellung)"]
    assert C.PRESETS["Schlechtes GPS"]["sz"] > base["sz"]
    long = C.PRESETS["Langer Tunnel, ohne Radsensor"]
    assert long["tun"] == C.TUNNEL_MAX and long["rad"] == "aus"
    assert C.PRESETS["Sehr kurze Verzögerung"]["lag"] < base["lag"]
