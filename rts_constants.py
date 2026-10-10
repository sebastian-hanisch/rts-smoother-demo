"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 120 Schritte
DT = 1.0
T_STEPS = 120
N_BURN = 10                       # Einschwingzeit: Kennzahlen zählen erst ab diesem Schritt

# feste Annahmen (gleiche Fahrt wie im Kalman-Filter-Stück: der Filter kennt hier die Wahrheit)
V_START_MEAN = 10.0               # mittlere Startgeschwindigkeit in m/s (36 km/h)
V_START_STD = 1.0
V_PRIOR_STD = 5.0
SIGMA_WHEEL = 0.3                 # Rauschen des Radsensors in m/s
TUNNEL_START = 50                 # Beginn des GPS-Ausfalls (Schritt)

# Reglergrenzen
SIGMA_Z_MIN, SIGMA_Z_MAX, DEFAULT_SIGMA_Z = 3.0, 30.0, 10.0
Q_MIN, Q_MAX, DEFAULT_Q = 0.05, 0.8, 0.3
TUNNEL_MIN, TUNNEL_MAX, DEFAULT_TUNNEL = 0, 40, 20
WHEEL_CHOICES = ("an", "aus")
DEFAULT_WHEEL = "an"
LAG_MIN, LAG_MAX, DEFAULT_LAG = 0, 40, 5
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100040))
SIGMA_SWEEP = (3.0, 6.0, 10.0, 15.0, 20.0, 30.0)
LAG_SWEEP = (0, 1, 2, 3, 5, 8, 12, 20, 40)

STATE_DIM = 2

COLOR_TRUTH = "#2F6B65"
COLOR_FILTER = "#c97b2a"
COLOR_SMOOTH = "#4c78a8"
COLOR_LAG = "#7a5fa6"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_BAND_FILTER = "rgba(201,123,42,0.14)"
COLOR_TUNNEL = "rgba(120,120,120,0.12)"

PRESETS = {
    "Standardfahrt (Voreinstellung)": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "tun": DEFAULT_TUNNEL, "rad": "an", "lag": DEFAULT_LAG, "seed": DEFAULT_SEED},
    "Schlechtes GPS": {"sz": 25.0, "q": DEFAULT_Q, "tun": DEFAULT_TUNNEL, "rad": "an", "lag": DEFAULT_LAG, "seed": DEFAULT_SEED},
    "Langer Tunnel, ohne Radsensor": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "tun": 40, "rad": "aus", "lag": DEFAULT_LAG, "seed": DEFAULT_SEED},
    "Sehr kurze Verzögerung": {"sz": DEFAULT_SIGMA_Z, "q": DEFAULT_Q, "tun": DEFAULT_TUNNEL, "rad": "an", "lag": 1, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfahrt (Voreinstellung)": "GPS mit 10 m Rauschen, 20 s Tunnel, Radsensor an, Glätter mit 5 s Verzögerung.",
    "Schlechtes GPS": "Das GPS rauscht mit 25 m statt 10 m: es gibt mehr zu glätten.",
    "Langer Tunnel, ohne Radsensor": "40 s ohne GPS und ohne Radsensor: der Filter driftet, der Glätter zieht die Lücke von beiden Seiten zusammen.",
    "Sehr kurze Verzögerung": "Der Glätter darf nur 1 s in die Zukunft sehen: wie viel vom Gewinn bleibt?",
}
