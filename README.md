# RTS-Glätter: die Schätzung im Nachhinein verbessern – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-rts-smoother-demo.streamlit.app/)** -->

Zweites Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt die Fahrt des Stücks [Kalman-Filter](https://github.com/sebastian-hanisch/kalman-filter-demo) fort (Lieferfahrzeug, GPS, Radsensor, Tunnel) und behebt eine benennbare
Schwäche des Filters: **er nutzt nur die Vergangenheit**. Der Rauch-Tung-Striebel-Glätter rechnet dieselbe Fahrt rückwärts und schätzt jeden Zeitpunkt aus allen Messungen.
Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Wer eine Fahrt **nachträglich** auswertet (Routen-Nachbereitung, Abrechnung, Training eines Modells), muss nicht auf Echtzeit Rücksicht nehmen. Die Kernfrage: **Was bringt der Blick
in die Zukunft, und wie viel Verzögerung braucht man, um davon zu profitieren?** Die zweite Hälfte betrifft den Mittelweg: Auswertung in Echtzeit, aber mit einigen Sekunden Verspätung.

## Modell

Dasselbe Modell wie im Kalman-Filter-Stück (Zustand Position und Geschwindigkeit, GPS und Radsensor, Joseph-Form), dazu die Rückwärtsrechnung
$C_k = P_{k|k}F^\top P_{k+1|k}^{-1}$, $\hat x_{k|T} = \hat x_{k|k} + C_k(\hat x_{k+1|T} - \hat x_{k+1|k})$ und der Verzögerungs-Glätter (Rückwärtsrechnung nur $L$ Schritte lang).
Formeln im 📐-Expander der App, Kern in `rts_algorithm.py`. Die Wahrheit wird mit dem Modell des Filters erzeugt: der Filter kennt die Rauschwerte.

## Methodik

Drei Experimente auf Knopfdruck, je über 40 feste Fahrten (`SWEEP_SEEDS`):

1. Positionsfehler Filter gegen Glätter bei GPS-Rauschen von 3 bis 30 m.
2. Verzögerung: Anteil des Gewinns, den ein fester Lag von 0 bis 40 s erreicht.
3. Tunnel: Fehler auf den Tunnelsekunden, mit und ohne Radsensor.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Standardeinstellung (GPS 10 m, Fahrunruhe 0,3 m/s², Tunnel 20 s, Radsensor an), Median über die 40 festen Fahrten; `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Wie viel genauer wird die Position? | Der Fehler sinkt um **29 %** bei 3 m GPS-Rauschen, **31 %** bei 10 m und **46 %** bei 30 m (Median der Verbesserung je Fahrt). |
| Wie viel Verzögerung braucht es? | Ein Lag von 5 s holt nur **21 %** des Gewinns, 20 s **71 %**, 40 s **85 %**. Kurze Verzögerungen bringen wenig. |
| Tunnel von 40 s ohne GPS und Radsensor | Filter **28 m** daneben (RMSE), Glätter **7 m**: er kennt den Ausgang des Tunnels. |
| Dasselbe mit Radsensor | **1,27 m** gegen **1,04 m**: die Geschwindigkeitsmessung trägt die Position ohnehin. |
| Einzelne Standardfahrt (Seed 35) | Filter 1,07 m, Glätter 0,85 m, 5-s-Lag-Glätter 0,79 m auf dem Fenster ohne die letzten 5 s. Eine einzelne Fahrt, keine Aussage über die Streuung. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Fahrt und Filter teilen dasselbe Modell und bekannte Rauschwerte; es ist kein echter GPS-Datensatz im Spiel.
- **Der Gewinn mit Radsensor ist mäßig.** Die Geschwindigkeitsmessung verhindert schon beim Filter die größte Drift; im Tunnel ohne Radsensor sind die Unterschiede groß, mit Radsensor klein.
- **Verzögerung ist teuer.** Ein Lag von wenigen Sekunden bringt wenig; wer mit Verzögerung echtzeitnah arbeiten will, braucht nach diesen Messungen Größenordnung zehn Sekunden und mehr, um den Hauptteil des Gewinns zu sehen. Die Erklärung über die kleine Verstärkung des Filters ist plausibel, aber hier nicht eigens gemessen.
- **Eine einzelne Fahrt kann anders ausgehen.** Auf einzelnen Fahrten ist der Glätter nicht immer besser; belastbar sind die Mediane über 40 Fahrten. Die mittlere NEES einer Einzelfahrt streut stark (Orakel über 150 Fahrten: 2,0).
- **Dieselben Annahmen wie der Filter.** Lineares Modell, gaußsches Rauschen, eine Hypothese: der Glätter behebt die Zukunftsblindheit, sonst nichts.

## Verifikation

- **Kern:** Rückwärtsschritt und Glätter-Verstärkung gegen Handrechnung (`tests/test_algorithm.py`), Randfälle (letzter Schritt = Filter, Unsicherheit wird nie größer, Lag 0 = Filter, Lag zu groß).
- **Orakel** (`tests/test_oracle_rts_smoother.py`, anderer Rechenweg): alle geglätteten Zustände **und** ihre Kovarianzen gegen die Lösung der ganzen Fahrt als gewichtetes Kleinste-Quadrate-System (mit GPS-Lücke, mit und ohne Radsensor); der Verzögerungs-Glätter gegen den vollen Glätter auf abgeschnittenen Daten; mittlere NEES des Glätters über 150 Fahrten gleich 2; die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, Radsensor, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `rts_algorithm.py` | Kern: Modell, Filter (Joseph-Form), Glätter-Verstärkung, RTS-Rückwärtsrechnung, Verzögerungs-Glätter |
| `rts_scenario.py` | Fahrt und Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `rts_evaluation.py` | Kennzahlen, Verdikte, Sweeps über die festen Fahrten |
| `rts_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `rts_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `rts_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `rts_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Nichtlineare Modelle und Ausreißer:** der Glätter erbt sie vom Filter; eigene Stücke der Linie.
- **Optimale feste Fenster und Rechenaufwand** über große Spuren: nur gezeigt, wie sich der Fehler mit dem Lag ändert.
- **Echte GPS-Spuren:** kein Datensatz eingebunden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
