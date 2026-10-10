"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen, Kennzahlen, Verdikt. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "RTS-Glätter: die Schätzung im Nachhinein verbessern", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("GPS-Rauschen (wahr)", f"{_num(s.sigma_z, 1)} m"),
        ("Beschleunigungsrauschen (wahr)", f"{_num(s.q, 2)} m/s²"),
        ("Tunnel", f"{s.tunnel} s"),
        ("Radsensor", s.wheel),
        ("Verzögerung des Lag-Glätters", f"{s.lag} s"),
        ("Zufalls-Seed", str(s.seed)),
    ]
    for k, v in rows:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Positionsfehler Filter (RMSE)", f"{_num(a.rmse_f)} m"),
        ("Positionsfehler Glätter (RMSE)", f"{_num(a.rmse_s)} m"),
        ("Positionsfehler Lag-Glätter (RMSE)", f"{_num(a.rmse_lag)} m"),
        ("Mittlere NEES Filter / Glätter (Soll 2)", f"{_num(a.nees_f)} / {_num(a.nees_s)}"),
        ("Anteil im 95-%-Band (Glätter)", f"{_num(100 * a.coverage_s, 1)} %"),
    ]
    for k, v in metrics:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Synthetische Fahrt mit bekanntem Modell; die Zahlen gelten für dieses Szenario und diese Einstellungen.")
    return bytes(pdf.output())
