"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import rts_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _tunnel_rect(fig, track):
    idx = np.where(track.tunnel)[0]
    if len(idx):
        fig.add_vrect(x0=track.t[idx[0]], x1=track.t[idx[-1]], fillcolor=C.COLOR_TUNNEL, line_width=0, annotation_text="Tunnel (kein GPS)",
                      annotation_position="top left")


def build_error_chart(a):
    """Positionsfehler gegen die Wahrheit: Filter (nur Vergangenheit) und Glätter (alle Daten), jeweils mit 95-%-Band."""
    t = a.track.t
    bf, bs = 1.96 * a.sd_f, 1.96 * a.sd_s
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[bf, -bf[::-1]], fill="toself", fillcolor=C.COLOR_BAND_FILTER, line=dict(width=0),
                             hoverinfo="skip", name="95-%-Band Filter"))
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[bs, -bs[::-1]], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                             hoverinfo="skip", name="95-%-Band Glätter"))
    fig.add_trace(go.Scatter(x=t, y=a.err_f, mode="lines", line=dict(color=C.COLOR_FILTER, width=2), name="Filter",
                             hovertemplate="t = %{x:.0f} s: %{y:.1f} m<extra>Filter</extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.err_s, mode="lines", line=dict(color=C.COLOR_SMOOTH, width=3), name="Glätter",
                             hovertemplate="t = %{x:.0f} s: %{y:.1f} m<extra>Glätter</extra>"))
    _tunnel_rect(fig, a.track)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Positionsfehler (m)")
    return _base(fig)


def build_band_chart(a):
    """Angegebene Unsicherheit (1 Sigma) der Position über die Zeit: Filter wächst im Tunnel, der Glätter überbrückt ihn von beiden Seiten."""
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.sd_f, mode="lines", line=dict(color=C.COLOR_FILTER, width=3), name="Filter"))
    fig.add_trace(go.Scatter(x=t, y=a.sd_s, mode="lines", line=dict(color=C.COLOR_SMOOTH, width=3), name="Glätter"))
    if a.settings.lag > 0:
        n = len(a.x_lag)
        fig.add_trace(go.Scatter(x=t[:n], y=np.sqrt(a.P_lag[:, 0, 0]), mode="lines", line=dict(color=C.COLOR_LAG, width=2, dash="dash"),
                                 name=f"Lag-Glätter ({a.settings.lag} s)"))
    _tunnel_rect(fig, a.track)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Unsicherheit der Position, 1 Sigma (m)")
    return _base(fig, 300)


def build_sigma_sweep(rows):
    xs = [r["sigma"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["filter"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_FILTER, width=3), name="Filter"))
    fig.add_trace(go.Scatter(x=xs, y=[r["smooth"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_SMOOTH, width=3), name="Glätter"))
    fig.update_xaxes(title_text="Wahres GPS-Rauschen (m)")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)


def build_lag_sweep(rows):
    xs = [r["lag"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["filter"] for r in rows], mode="lines", line=dict(color=C.COLOR_FILTER, width=2, dash="dot"), name="Filter (nur Vergangenheit)"))
    fig.add_trace(go.Scatter(x=xs, y=[r["smooth"] for r in rows], mode="lines", line=dict(color=C.COLOR_SMOOTH, width=2, dash="dot"), name="Glätter über die ganze Fahrt"))
    fig.add_trace(go.Scatter(x=xs, y=[r["rmse"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LAG, width=3), name="Glätter mit Verzögerung"))
    fig.update_xaxes(title_text="Erlaubte Verzögerung (s)")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)


def build_tunnel_sweep(rows):
    xs = [r["tunnel"] for r in rows]
    fig = go.Figure()
    for key, label, color, dash in (("f_aus", "Filter ohne Radsensor", C.COLOR_FILTER, "dash"), ("s_aus", "Glätter ohne Radsensor", C.COLOR_SMOOTH, "dash"),
                                    ("f_an", "Filter mit Radsensor", C.COLOR_FILTER, "solid"), ("s_an", "Glätter mit Radsensor", C.COLOR_SMOOTH, "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=3, dash=dash), name=label))
    fig.update_xaxes(title_text="Tunnellänge (s)")
    fig.update_yaxes(title_text="Positionsfehler im Tunnel, RMSE (m)")
    return _base(fig)
