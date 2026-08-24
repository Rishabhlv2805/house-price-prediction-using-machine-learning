"""House price prediction — Streamlit studio (paper / ink)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "reports" / "metrics.json"

GITHUB_REPO = "https://github.com/Rishabhlv2805/house-price-prediction-using-machine-learning"
GITHUB_USER = "https://github.com/Rishabhlv2805"

PAPER = "#F3EEE4"
CARD = "#FCFAF6"
SURFACE = "#EBE4D6"
INK = "#1C1915"
MUTED = "#6F675C"
ACCENT = "#3E5363"
GRID = "rgba(28,25,21,0.12)"
USD = 100_000

APP_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap");

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
  background: #F3EEE4 !important;
  color: #1C1915;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
footer, #MainMenu, .stDeployButton, [data-testid="stDeployButton"] { display: none !important; }
iframe[height="0"], iframe[height="0px"] { display: none !important; }

h1, h2, h3, .serif, .wordmark, .hero, .hero-num, .stat-card .value, .insight h3 {
  font-family: "Instrument Serif", Georgia, serif;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: #1C1915;
}
p, label, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] {
  font-family: "Figtree", Helvetica, sans-serif;
}

.kicker { color: #6F675C; letter-spacing: 0.18em; font-size: 0.72rem; text-transform: uppercase; font-weight: 500; margin: 0; }
.hero { font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.08; margin: 0.35rem 0 0.8rem; }
.lede { color: #6F675C; font-size: 1.05rem; max-width: 36rem; }
.muted { color: #6F675C; }

.scorecard { background: #EBE4D6; border: 1px solid rgba(28,25,21,0.12); border-radius: 16px; padding: 1.4rem 1.5rem; }
.hero-num { font-size: 4.4rem; line-height: 0.9; margin: 0.6rem 0 0.2rem; }
.stat-card { background: #FCFAF6; border: 1px solid rgba(28,25,21,0.12); border-radius: 12px; padding: 0.95rem 1.05rem; }
.stat-card .value { font-size: 1.7rem; margin: 0.2rem 0 0; }
.insight { background: #FCFAF6; border: 1px solid rgba(28,25,21,0.12); border-radius: 12px; padding: 1.1rem 1.15rem; height: 100%; }
.insight h3 { font-size: 1.15rem; margin: 0 0 0.4rem; }

.wordmark { font-size: 1.85rem; margin: 0; }
.author { color: #6F675C; font-size: 0.85rem; margin: 0; text-align: right; }
.author a { color: #1C1915; text-decoration: none; border-bottom: 1px solid rgba(28,25,21,0.25); }

div[data-testid="stMetric"] { background: #FCFAF6; border: 1px solid rgba(28,25,21,0.12); border-radius: 12px; padding: 0.8rem 1rem; }
.stRadio [role="radiogroup"] { background: #EBE4D6; padding: 0.25rem; border-radius: 10px; }
.stRadio [data-baseweb="radio"] { padding: 0.35rem 0.85rem; }
"""


def inject_css() -> None:
    """Write CSS into the parent document. Streamlit sanitizes <style> in markdown."""
    components.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          let style = doc.getElementById("hp-css");
          if (!style) {{
            style = doc.createElement("style");
            style.id = "hp-css";
            doc.head.appendChild(style);
          }}
          style.textContent = {json.dumps(APP_CSS)};
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


@st.cache_data
def load_metrics() -> dict:
    with METRICS_PATH.open() as f:
        return json.load(f)


def money(value: float) -> str:
    return f"${value:,.0f}"


def paper_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD,
        font=dict(color=INK, family="Figtree, Helvetica, sans-serif", size=13),
        height=height,
        margin=dict(l=40, r=20, t=16, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
        colorway=[ACCENT, INK, "#8A8276"],
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def predict_linear(metrics: dict, features: dict[str, float]) -> tuple[float, list[tuple[str, float]]]:
    lin = metrics["linear"]
    units = float(lin["intercept"])
    parts: list[tuple[str, float]] = []
    labels = metrics["feature_labels"]
    for col in metrics["features"]:
        mean = float(lin["scaler_mean"][col])
        scale = float(lin["scaler_scale"][col]) or 1.0
        z = (float(features[col]) - mean) / scale
        contribution = z * float(lin["coefficients"][col])
        units += contribution
        parts.append((labels.get(col, col), contribution))
    parts.sort(key=lambda x: abs(x[1]), reverse=True)
    return units, parts


st.set_page_config(
    page_title="House Prices · California Housing",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

m = load_metrics()
d = m["dataset"]
winner_name = m["winner"]
winner = next(r for r in m["models"] if r["winner"])
linear = next(r for r in m["models"] if r["name"] == "Linear Regression")
importance_map = {row["feature"]: float(row["importance"]) for row in m["importance"]}
income_pct = importance_map.get("MedInc", 0) * 100
occupancy_pct = importance_map.get("AveOccup", 0) * 100
geo_pct = (importance_map.get("Latitude", 0) + importance_map.get("Longitude", 0)) * 100
test_n = int(d.get("test_size", 4128))

brand, nav, byline = st.columns([1.1, 2.2, 1.2], vertical_alignment="center")
with brand:
    st.markdown('<p class="wordmark">House Prices</p>', unsafe_allow_html=True)
with nav:
    page = st.radio(
        "Navigate",
        ["Studio", "Data", "Models", "Estimate"],
        horizontal=True,
        label_visibility="collapsed",
    )
with byline:
    st.markdown(
        f'<p class="author" style="text-align:right">Rishabh Sharma · '
        f'<a href="{GITHUB_REPO}">GitHub</a></p>',
        unsafe_allow_html=True,
    )

st.divider()


if page == "Studio":
    left, right = st.columns([1.25, 0.85], gap="large")
    with left:
        st.markdown('<p class="kicker">California Housing · 20,640 districts</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero">What is a house worth?</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="lede">Four regression models trained on the same 80/20 split. '
            f'<strong>{winner_name}</strong> is the holdout champion at R² {winner["r2"]:.3f}.</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="scorecard">
              <p class="kicker">Best model · {winner_name}</p>
              <p class="hero-num">{winner["r2"]:.3f}</p>
              <p class="muted">Test R²</p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1.2rem">
                <div><p class="muted" style="margin:0">MAE</p><p style="margin:0;font-variant-numeric:tabular-nums">{money(winner["mae_usd"])}</p></div>
                <div><p class="muted" style="margin:0">RMSE</p><p style="margin:0;font-variant-numeric:tabular-nums">{money(winner["rmse_usd"])}</p></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    s1, s2, s3, s4 = st.columns(4)
    for col, label, value in (
        (s1, "Districts", f"{d['n_samples']:,}"),
        (s2, "Features", str(d["n_features"])),
        (s3, "Train / test", "80 / 20"),
        (s4, "Linear R²", f"{linear['r2']:.4f}"),
    ):
        col.markdown(
            f'<div class="stat-card"><p class="kicker">{label}</p><p class="value">{value}</p></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2, gap="large")
    res = pd.DataFrame(m["models"])
    with c1:
        st.subheader("Holdout RMSE")
        st.caption(f"Typical error on {test_n:,} unseen block groups. Lower is better.")
        fig = go.Figure(
            go.Bar(
                x=res["name"],
                y=res["rmse_usd"],
                marker_color=[ACCENT if w else INK for w in res["winner"]],
                hovertemplate="%{x}<br>RMSE %{y:$,.0f}<extra></extra>",
            )
        )
        fig.update_layout(yaxis_title="USD", showlegend=False)
        st.plotly_chart(paper_layout(fig, 340), use_container_width=True)
    with c2:
        st.subheader("What the model listens to")
        st.caption(
            f"XGBoost importance. Median income is {income_pct:.0f}% of the signal; "
            f"latitude and longitude together are another {geo_pct:.0f}%."
        )
        fi = pd.DataFrame(m["importance"])
        fi["label"] = fi["feature"].map(m["feature_labels"])
        fig = go.Figure(
            go.Bar(
                x=fi["importance"][::-1],
                y=fi["label"][::-1],
                orientation="h",
                marker_color=ACCENT,
                hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title="Importance", showlegend=False)
        st.plotly_chart(paper_layout(fig, 340), use_container_width=True)

    st.subheader("Scorecard")
    st.caption("Metrics on the held-out 20%. Trees do not use the scaler; linear regression does.")
    table = pd.DataFrame(
        [
            {
                "Model": row["name"] + ("  · winner" if row["winner"] else ""),
                "MAE": money(row["mae_usd"]),
                "RMSE": money(row["rmse_usd"]),
                "R²": f"{row['r2']:.4f}",
            }
            for row in m["models"]
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    i1, i2, i3 = st.columns(3)
    insights = [
        (
            "Income dominates",
            f"Median block-group income is {income_pct:.1f}% of XGBoost importance. Ability to pay is still the loudest price signal in the state.",
        ),
        (
            "Place is the rest",
            f"Longitude and latitude together are {geo_pct:.0f}%. Coastal and Bay Area premiums survive after income is accounted for.",
        ),
        (
            "The $500k ceiling",
            "The target is censored at $500,000. Residuals fan out at the top of the market because the labels themselves are clipped.",
        ),
    ]
    for col, (title, body) in zip((i1, i2, i3), insights):
        col.markdown(
            f'<div class="insight"><h3>{title}</h3><p class="muted" style="margin:0">{body}</p></div>',
            unsafe_allow_html=True,
        )


elif page == "Data":
    st.markdown('<p class="kicker">Exploratory analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero">20,640 block groups, zero missing cells</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">The California Housing set from scikit-learn is a 1990 census extract. '
        "The target is median house value in units of $100,000, censored at $500,000. "
        "Outlier occupancy and room ratios were clipped using training-set 1st/99th percentiles.</p>",
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2, gap="large")
    hist = m["charts"]["target_hist"]
    centers = [(hist["edges"][i] + hist["edges"][i + 1]) / 2 for i in range(len(hist["counts"]))]
    with g1:
        st.subheader("Target distribution")
        st.caption("Units of $100,000. The spike at 5.0 is the $500k cap, not a pile of identical homes.")
        fig = go.Figure(
            go.Bar(
                x=centers,
                y=hist["counts"],
                marker_color=ACCENT,
                hovertemplate="Value %{x:.2f} × $100k<br>%{y} districts<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title="Median house value ($100k)", yaxis_title="Block groups")
        st.plotly_chart(paper_layout(fig, 360), use_container_width=True)
    with g2:
        st.subheader("Income versus value")
        st.caption("The strongest linear relationship in the table. Scatter is a sample so the page stays light.")
        inc = pd.DataFrame(m["charts"]["income"])
        fig = go.Figure(
            go.Scatter(
                x=inc["x"],
                y=inc["y"],
                mode="markers",
                marker=dict(size=5, color=ACCENT, opacity=0.4),
                hovertemplate="Income %{x:.2f}<br>Value %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title="Median income ($10k)", yaxis_title="Median value ($100k)")
        st.plotly_chart(paper_layout(fig, 360), use_container_width=True)

    st.subheader("Where the expensive lots sit")
    st.caption("Darker marks are higher median values. The coast and the Bay are visible without a basemap.")
    geo = pd.DataFrame(m["charts"]["geo"])
    fig = go.Figure(
        go.Scatter(
            x=geo["lon"],
            y=geo["lat"],
            mode="markers",
            marker=dict(
                size=5,
                color=geo["price"],
                colorscale=[[0, "#EBE4D6"], [0.5, "#6F8A96"], [1, "#3E5363"]],
                opacity=0.8,
                colorbar=dict(title="$100k", thickness=12),
            ),
            hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.1)
    st.plotly_chart(paper_layout(fig, 480), use_container_width=True)

    st.subheader("Feature dictionary")
    rows = [
        {
            "Feature": col,
            "Meaning": m["feature_labels"][col],
            "Unit": m["feature_units"][col],
            "Train mean": m["feature_stats"][col]["mean"],
        }
        for col in m["features"]
    ]
    st.dataframe(pd.DataFrame(rows).style.format({"Train mean": "{:.3f}"}), use_container_width=True, hide_index=True)

    st.subheader("Correlation")
    corr = m["charts"]["correlation"]
    heat = go.Figure(
        go.Heatmap(
            z=corr["matrix"],
            x=corr["columns"],
            y=corr["columns"],
            colorscale=[[0, "#F3EEE4"], [0.5, "#C5B9A4"], [1, "#3E5363"]],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr["matrix"]],
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    st.plotly_chart(paper_layout(heat, 520), use_container_width=True)


elif page == "Models":
    st.markdown('<p class="kicker">Holdout evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero">Four models, one test fold</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="lede">Linear regression sees StandardScaler features. Trees see the winsorized but unscaled matrix. '
        f"Every score below is on the same {test_n:,}-row holdout. No test information entered preprocessing.</p>",
        unsafe_allow_html=True,
    )

    show = pd.DataFrame(
        [
            {
                "Model": row["name"] + ("  · winner" if row["winner"] else ""),
                "MAE": money(row["mae_usd"]),
                "MSE": f"{row['mse']:.3f}",
                "RMSE": money(row["rmse_usd"]),
                "R²": f"{row['r2']:.4f}",
            }
            for row in m["models"]
        ]
    )
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.success(
        f"Winner: **{winner_name}** — RMSE {money(winner['rmse_usd'])}, R² {winner['r2']:.4f}. "
        f"Linear RMSE was {money(linear['rmse_usd'])}."
    )

    avp = pd.DataFrame(m["charts"]["actual_vs_predicted"])
    resid = pd.DataFrame(m["charts"]["residuals"])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.subheader("Actual versus predicted")
        st.caption(f"{winner_name} on the test set. The diagonal is a perfect match. The ceiling at 5.0 is the dataset cap.")
        fig = go.Figure()
        fig.add_scatter(
            x=avp["actual"],
            y=avp["predicted"],
            mode="markers",
            marker=dict(size=5, color=ACCENT, opacity=0.35),
            name="Test points",
            hovertemplate="Actual %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>",
        )
        fig.add_scatter(
            x=[0, 5.2],
            y=[0, 5.2],
            mode="lines",
            name="Perfect fit",
            line=dict(dash="dash", color=MUTED),
        )
        fig.update_layout(xaxis_title="Actual ($100k)", yaxis_title="Predicted ($100k)")
        st.plotly_chart(paper_layout(fig, 420), use_container_width=True)
    with g2:
        st.subheader("Residuals versus predicted")
        st.caption("Errors fan out as predictions approach the $500k label cap.")
        fig = go.Figure(
            go.Scatter(
                x=resid["predicted"],
                y=resid["residual"],
                mode="markers",
                marker=dict(size=5, color=INK, opacity=0.3),
                hovertemplate="Predicted %{x:.2f}<br>Residual %{y:.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        fig.update_layout(xaxis_title="Predicted ($100k)", yaxis_title="Residual ($100k)")
        st.plotly_chart(paper_layout(fig, 420), use_container_width=True)

    st.markdown(
        f"**Why {winner_name} won.** House prices here mix smooth income gradients with sharp geographic breaks. "
        f"A linear plane cannot represent the Bay versus the Central Valley with the same slope, which is why RMSE "
        f"falls from {money(linear['rmse_usd'])} (linear) to {money(winner['rmse_usd'])}."
    )


elif page == "Estimate":
    st.markdown('<p class="kicker">Live estimator</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero">Price a district</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">The published champion is XGBoost. This page runs the linear model so every slider '
        "has an additive, inspectable effect. Values are median house prices for a census block group, not a listing.</p>",
        unsafe_allow_html=True,
    )

    preset_names = ["Average district"] + [p["name"] for p in m["presets"]]
    choice = st.radio("Region preset", preset_names, horizontal=True)

    if choice == "Average district":
        base = {col: float(m["feature_stats"][col]["mean"]) for col in m["features"]}
        actual = None
    else:
        found = next(p for p in m["presets"] if p["name"] == choice)
        base = {k: float(v) for k, v in found["features"].items()}
        actual = float(found["actual"])

    features: dict[str, float] = {}
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("District levers")
        st.caption("Ranges follow the training fold. Drag a slider, or pick a region above.")
        cols = st.columns(2)
        for i, col in enumerate(m["features"]):
            stat = m["feature_stats"][col]
            with cols[i % 2]:
                features[col] = st.slider(
                    f"{m['feature_labels'][col]}",
                    min_value=float(stat["min"]),
                    max_value=float(stat["max"]),
                    value=float(base[col]),
                    step=(float(stat["max"]) - float(stat["min"])) / 200,
                    help=m["feature_units"][col],
                )

    units, parts = predict_linear(m, features)
    usd = max(0.0, units * USD)

    with right:
        st.markdown(
            f"""
            <div class="scorecard">
              <p class="kicker">Linear estimate</p>
              <p class="hero-num" style="font-size:3rem">{money(usd)}</p>
              <p class="muted">{units:.2f} × $100,000 units</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if actual is not None:
            st.caption(f"Recorded median for this {choice} sample: {money(actual * USD)}")

        st.subheader("Feature contributions")
        drv = pd.DataFrame(parts, columns=["feature", "contribution"])
        fig = go.Figure(
            go.Bar(
                x=drv["contribution"][::-1],
                y=drv["feature"][::-1],
                orientation="h",
                marker_color=[(ACCENT if v > 0 else "#8A8276") for v in drv["contribution"][::-1]],
                hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title="Positive = higher predicted value")
        st.plotly_chart(paper_layout(fig, 360), use_container_width=True)
