"""RTS-Glätter: die Schätzung im Nachhinein verbessern - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Fahrt wie im Kalman-Filter-Stück (Lieferfahrzeug, GPS, Radsensor, Tunnel). Der Filter nutzt nur die Vergangenheit; der
Rauch-Tung-Striebel-Glätter rechnet rückwärts und nutzt auch die Zukunft. Dazu: wie viel Verzögerung für den Gewinn nötig ist.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import rts_constants as C
import rts_evaluation as ev
from rts_evaluation import Settings, analyse, lag_share
from rts_pdf_export import build_pdf
from rts_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rts_visualization import build_band_chart, build_error_chart, build_lag_sweep, build_sigma_sweep, build_tunnel_sweep

st.set_page_config(page_title="RTS-Glätter – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_sigma(settings):
    return ev.sweep_sigma(settings)


@st.cache_data(show_spinner=False)
def _sweep_lag(settings):
    return ev.sweep_lag(settings)


@st.cache_data(show_spinner=False)
def _sweep_tunnel(settings):
    return ev.sweep_tunnel(settings)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (wenige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner wenige Sekunden)"):
        return fn(settings)


st.title("⏪ RTS-Glätter: die Schätzung im Nachhinein verbessern")
st.markdown(
    """
Der **Kalman-Filter** kennt in jeder Sekunde nur die **Vergangenheit**: das ist richtig für Echtzeit, aber verschenkt Information, wenn man später auswertet.
Der **Rauch-Tung-Striebel-Glätter** (RTS) rechnet dieselbe Fahrt **rückwärts** noch einmal und schätzt jeden Zeitpunkt aus **allen** Messungen, auch den späteren.
Die Demo ist das zweite Stück der Linie **Zustandsschätzung und Filter** und setzt die Fahrt des Kalman-Filter-Stücks fort. Im Unterschied zu den Fall-Demos zeigt sie ein einziges
Verfahren an einem wachsenden Beispiel.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der RTS-Glätter", expanded=True):
    st.markdown(
        """
1. **Vorwärts:** ein gewöhnlicher Kalman-Filter läuft über die ganze Fahrt und merkt sich dabei Vorhersage und Korrektur jeder Sekunde.
2. **Rückwärts:** von der letzten Sekunde an wird jede Schätzung korrigiert: Was die **spätere** Schätzung von der **Vorhersage** abweicht, wirkt auf die frühere Sekunde zurück, mit
   einem Gewicht, das aus der Unsicherheit folgt.
3. **Ergebnis:** zu jeder Sekunde eine Schätzung **und** eine Unsicherheit, die höchstens so groß ist wie die des Filters.

**Warum das funktioniert:** Zwei benachbarte Zeitpunkte hängen über das Fahrzeugmodell zusammen. Eine Messung in Sekunde 70 sagt auch etwas über Sekunde 60,
weil das Fahrzeug dazwischen nur begrenzt beschleunigen kann.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Fahrt**")
    sigma_z = st.slider("GPS-Rauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=0.5,
                        help="Streuung der GPS-Position um die wahre Position (Standardabweichung, in Metern).")
    q = st.slider("Fahrunruhe (m/s²)", *bounds("q_slider"), key="q_slider", step=0.05,
                  help="Wie stark die Beschleunigung zufällig schwankt.")
    tunnel = st.slider("Tunnellänge (s)", *bounds("tunnel_slider"), key="tunnel_slider", step=5,
                       help="So lange fällt das GPS ab Sekunde 50 aus. Null heißt: kein Tunnel.")
    wheel = st.radio("Radsensor", C.WHEEL_CHOICES, key="wheel_select", horizontal=True,
                     help="Der Radsensor misst die Geschwindigkeit (0,3 m/s Rauschen) und läuft auch im Tunnel weiter.")
    st.markdown("**Die Auswertung**")
    lag = st.slider("Erlaubte Verzögerung (s)", *bounds("lag_slider"), key="lag_slider",
                    help="Der Verzögerungs-Glätter darf so viele Sekunden in die Zukunft sehen. 0 ist der Filter, die ganze Fahrt ist der volle Glätter.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sigma_slider": float(sigma_z), "q_slider": float(q), "tunnel_slider": int(tunnel), "wheel_select": wheel,
                   "lag_slider": int(lag), "seed_input": int(seed)})

settings = Settings(float(sigma_z), float(q), int(tunnel), wheel, int(lag), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Positionsfehler Glätter", f"{de(a.rmse_s, 2)} m", delta=f"{a.rmse_s - a.rmse_f:+.2f} m gegen Filter".replace(".", ","), delta_color="inverse",
          help="Wurzel des mittleren quadratischen Fehlers (RMSE) über alle Sekunden ab Sekunde 10, auch im Tunnel.")
m2.metric("Positionsfehler Filter", f"{de(a.rmse_f, 2)} m", delta="Referenz (nur Vergangenheit)", delta_color="off")
m3.metric(f"Glätter mit {settings.lag} s Verzögerung", f"{de(a.rmse_lag, 2)} m",
          delta=f"{a.rmse_lag - a.rmse_f_w:+.2f} m gegen Filter, gleiches Fenster".replace(".", ","), delta_color="inverse",
          help="Gleiche Zeitpunkte wie der Vergleichswert: die letzten Sekunden der Fahrt hat dieser Glätter noch nicht gesehen.")
m4.metric("Mittlere NEES Glätter (Soll 2)", de(a.nees_s, 2), delta=f"Filter: {de(a.nees_f, 2)}", delta_color="off",
          help="Normiertes Fehlerquadrat: passt der wahre Fehler zur angegebenen Unsicherheit? Soll: die Zustandsdimension, hier 2.")

st.markdown("**Positionsfehler über die Zeit** – orange: Filter, blau: Glätter, Bänder: das 95-%-Intervall, das der jeweilige Schätzer sich selbst zuschreibt.")
st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
st.markdown("**Angegebene Unsicherheit** – der Glätter ist nie unsicherer als der Filter; im Tunnel schließt er die Lücke von beiden Seiten.")
st.plotly_chart(build_band_chart(a), width="stretch", key="band_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="rts_glaetter_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was bringt der Blick in die Zukunft?")
gain = 1.0 - a.rmse_s / a.rmse_f if a.rmse_f > 0 else 0.0
band_gain = 1.0 - float(np.mean(a.sd_s[C.N_BURN:])) / float(np.mean(a.sd_f[C.N_BURN:]))
if gain >= 0.15:
    st.success(
        f"Der Glätter ist auf dieser Fahrt um **{100 * gain:.0f} %** genauer als der Filter ({de(a.rmse_s, 2)} statt {de(a.rmse_f, 2)} m), "
        f"und sein angegebenes Band ist im Mittel {100 * band_gain:.0f} % schmaler."
    )
elif gain > 0:
    st.info(f"Der Glätter ist nur **{100 * gain:.0f} %** genauer ({de(a.rmse_s, 2)} statt {de(a.rmse_f, 2)} m): mit Radsensor und wenig Rauschen gibt es kaum etwas nachzubessern.")
else:
    st.warning(
        f"Auf **dieser** Fahrt ist der Glätter nicht besser ({de(a.rmse_s, 2)} gegen {de(a.rmse_f, 2)} m). Im Mittel ist er es (Experiment 1), "
        "einzelne Fahrten können anders ausgehen, weil der Fehler einer Fahrt Zufall ist."
    )
st.caption("Eine einzelne Fahrt streut; die Experimente unten mitteln über 40 feste Fahrten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie viel genauer wird die Schätzung, und wann?")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Fahrunruhe, Tunnel und Radsensor wie in der Seitenleiste, das GPS-Rauschen läuft von 3 bis 30 m.")
rows = run_experiment("sigma", _sweep_sigma, settings, f"▶ Experiment 1 rechnen ({len(C.SIGMA_SWEEP) * len(C.SWEEP_SEEDS)} Fahrten)")
if rows is not None:
    st.plotly_chart(build_sigma_sweep(rows), width="stretch", key="sigma_chart")
    lo, hi = rows[0], rows[-1]
    st.info(
        f"Bei {lo['sigma']:.0f} m GPS-Rauschen verringert der Glätter den Positionsfehler im Median um **{100 * lo['gain']:.0f} %**, bei {hi['sigma']:.0f} m um "
        f"**{100 * hi['gain']:.0f} %**. Je verrauschter die Messung, desto mehr gibt es nachträglich zu mitteln."
    )

st.subheader("🔬 Experiment 2: Wie viel Verzögerung braucht man für den Gewinn?")
st.caption(f"Fester Lag: der Schätzer sieht für jeden Zeitpunkt die folgenden L Sekunden. Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Fenster je Lag gleich für alle drei Linien.")
lag_rows = run_experiment("lag", _sweep_lag, settings, f"▶ Experiment 2 rechnen ({len(C.LAG_SWEEP) * len(C.SWEEP_SEEDS)} Auswertungen)")
if lag_rows is not None:
    st.plotly_chart(build_lag_sweep(lag_rows), width="stretch", key="lag_chart")
    parts = []
    for L in (5, 20, 40):
        sh = lag_share(lag_rows, L)
        if sh is not None:
            parts.append(f"{L} s Verzögerung: {100 * sh:.0f} %")
    st.info(
        "Anteil des Gewinns (Filter → voller Glätter), den ein fester Lag erreicht: " + "; ".join(parts) + ". "
        "Eine naheliegende Erklärung (hier nicht eigens gemessen): bei kleinem Gewicht der einzelnen Messung, wie im ersten Stück gezeigt, wirkt jede Messung lange nach, "
        "und der Glätter braucht entsprechend lange, bis er davon profitiert."
    )

st.subheader("🔬 Experiment 3: Und im Tunnel?")
st.caption(f"Positionsfehler nur auf den Tunnelsekunden (Median über {len(C.SWEEP_SEEDS)} feste Fahrten), mit und ohne Radsensor.")
trows = run_experiment("tunnel", _sweep_tunnel, settings, f"▶ Experiment 3 rechnen ({4 * 2 * len(C.SWEEP_SEEDS)} Fahrten)")
if trows is not None:
    st.plotly_chart(build_tunnel_sweep(trows), width="stretch", key="tunnel_chart")
    t = trows[-1]
    st.info(
        f"Bei {t['tunnel']} s ohne GPS und ohne Radsensor liegt der Filter im Tunnel um {de(t['f_aus'])} m daneben (RMSE, Median über die Fahrten), der Glätter nur um {de(t['s_aus'])} m: "
        "er kennt den Ausgang des Tunnels. Mit Radsensor ist die Drift ohnehin klein "
        f"({de(t['f_an'], 2)} gegen {de(t['s_an'], 2)} m)."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es wird nachträglich ausgewertet** | In Echtzeit ist die Zukunft nicht verfügbar. Mit fester Verzögerung geht es, aber der Gewinn kommt erst mit viel Verzögerung (Experiment 2). | – |
| **Dasselbe lineare Gauß-Modell wie der Filter** | Der Glätter erbt alle Schwächen des Filters: falsches Rauschen macht auch ihn überselbstsicher, Ausreißer ziehen auch die Vergangenheit mit. | Erweiterte und robuste Filter |
| **Eine Hypothese** | Zwei mögliche Straßen glättet er zu einem Mittelwert dazwischen. | Hidden-Markov-Modelle, Partikelfilter |
| **Alles im Speicher** | Der volle Glätter braucht die ganze Fahrt samt Kovarianzen; bei langen Spuren sind das viele Zwischenwerte. | Fenster mit fester Verzögerung |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Vorwärts.** Kalman-Filter wie im ersten Stück: gefilterter Zustand $\hat x_{k|k}$, Kovarianz $P_{k|k}$, Vorhersage $\hat x_{k+1|k} = F\hat x_{k|k}$, $P_{k+1|k} = FP_{k|k}F^\top + Q$.

**Rückwärts** (Rauch-Tung-Striebel), für $k = T-2, \dots, 0$:
$$C_k = P_{k|k} F^\top P_{k+1|k}^{-1},\qquad
\hat x_{k|T} = \hat x_{k|k} + C_k\,(\hat x_{k+1|T} - \hat x_{k+1|k}),\qquad
P_{k|T} = P_{k|k} + C_k\,(P_{k+1|T} - P_{k+1|k})\,C_k^\top.$$

**Gaußsche Optimalität.** Bei linearem Modell und gaußschem Rauschen ist $\hat x_{k|T}$ der Erwartungswert des Zustands zum Zeitpunkt $k$ gegeben **alle** Messungen. Dasselbe
Ergebnis liefert die Lösung der gesamten Fahrt als gewichtetes Kleinste-Quadrate-System über alle Zustände gleichzeitig; die Demo prüft das gegen diese Rechnung.

**Verzögerungs-Glätter.** Der Zustand zum Zeitpunkt $k$ aus den Messungen bis $k + L$: dieselbe Rückwärtsrechnung, aber nur $L$ Schritte lang, gestartet bei $\hat x_{k+L|k+L}$.
$L = 0$ ist der Filter, $L = T-1$ der volle Glätter.

**Konsistenz.** $\mathrm{NEES}_k = (x_k - \hat x_{k|T})^\top P_{k|T}^{-1}(x_k - \hat x_{k|T})$ hat bei ehrlicher Schätzung den Erwartungswert 2.

**Literatur.** H. E. Rauch, F. Tung, C. T. Striebel (1965), *Maximum Likelihood Estimates of Linear Dynamic Systems*, AIAA Journal 3(8), 1445–1450.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
