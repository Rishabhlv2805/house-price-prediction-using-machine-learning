"""House price prediction — Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "reports" / "metrics.json"

GITHUB_REPO = "https://github.com/Rishabhlv2805/house-price-prediction-using-machine-learning"
GITHUB_USER = "https://github.com/Rishabhlv2805"

ACCENT = "#7dd3c0"
INK = "#e8eaed"
MUTED = "#9aa3b2"
GRID = "#2a2f3a"
USD = 100_000


@st.cache_data
def load_metrics() -> dict:
    with METRICS_PATH.open() as f:
        return json.load(f)


def dark_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Source Serif 4, Georgia, serif", size=13),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[ACCENT, "#c4b5a0", "#8ea0c4", "#d4a574", "#7aa2a8"],
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def money(value: float) -> str:
    return f"${value:,.0f}"


def predict_linear(metrics: dict, features: dict[str, float]) -> tuple[float, list[tuple[str, float]]]:
    lin = metrics["linear"]
    intercept = float(lin["intercept"])
    units = intercept
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
    page_title="House Price Prediction · California Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0f1115; }
      [data-testid="stHeader"] { background: rgba(15,17,21,0.9); }
      [data-testid="stSidebar"] { background: #12151c; border-right: 1px solid #2a2f3a; }
      h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif; letter-spacing: -0.02em; }
      .hero { font-size: 2.4rem; line-height: 1.15; margin: 0.2rem 0 0.8rem; }
      .kicker { color: #7dd3c0; letter-spacing: 0.18em; font-size: 0.72rem; text-transform: uppercase; }
      .muted { color: #9aa3b2; }
      .metric-card { background: #1a1d24; border: 1px solid #2a2f3a; border-radius: 14px; padding: 1rem 1.1rem; }
      div[data-testid="stMetric"] { background: #1a1d24; border: 1px solid #2a2f3a; border-radius: 14px; padding: 0.8rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

m = load_metrics()
d = m["dataset"]
winner_name = m["winner"]
winner = next(r for r in m["models"] if r["winner"])
linear = next(r for r in m["models"] if r["name"] == "Linear Regression")

with st.sidebar:
    st.markdown("**House Price Prediction**")
    st.caption("CALIFORNIA HOUSING · RISHABH SHARMA")
    page = st.radio(
        "Navigate",
        ["Overview", "Dataset", "Models", "Features", "Predict"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"[GitHub repo]({GITHUB_REPO})")
    st.markdown(f"[@Rishabhlv2805]({GITHUB_USER})")
    st.caption(f"{winner_name} · R² {winner['r2']:.3f} · {d['n_samples']:,} districts")


if page == "Overview":
    st.markdown('<p class="kicker">Machine learning project</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero">What is a California house worth?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Four regression models trained on the public California Housing set "
        f"({d['n_samples']:,} districts). **{winner_name}** leads on holdout RMSE "
        f"after an 80/20 split. Trees see unscaled features; linear regression uses a train-only scaler."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", f"{d['n_samples']:,}")
    c2.metric("Features", f"{d['n_features']}")
    c3.metric("Best model (R²)", winner_name)
    c4.metric("Best R²", f"{winner['r2']:.3f}")

    st.subheader("Why this problem")
    st.write(
        "Median block-group value is a mix of income, occupancy, and location. "
        "This lab trains four regressors on the same split so the models can be "
        "compared fairly, then inspects which levers actually move the price."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**What I did**")
        st.write(
            "- Audited missing values (none) and the $500k target cap\n"
            "- Winsorized occupancy / room outliers on the **train** fold only\n"
            "- Trained Linear Regression, Decision Tree, Random Forest, XGBoost\n"
            "- Compared MAE, MSE, RMSE, and R² on 4,128 held-out districts"
        )
    with right:
        st.markdown("**What stood out**")
        st.write(
            f"- **{winner_name}** RMSE {money(winner['rmse_usd'])} vs linear {money(linear['rmse_usd'])}\n"
            "- Median income is **37.8%** of XGBoost importance\n"
            "- Latitude + longitude together are ~**26%**\n"
            "- Residuals fan out at the $500k label censor"
        )

elif page == "Dataset":
    st.header("Dataset")
    st.write(
        f"**{d['name']}** · {d['n_samples']:,} rows × {d['n_features']} features · "
        f"target `{d['target']}` in units of ${d['usd_per_unit']:,}"
    )
    st.markdown("[scikit-learn source](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)")

    a, b, c = st.columns(3)
    a.metric("Missing cells", "0")
    b.metric("Train / test", "80 / 20")
    c.metric("Target cap", "$500,000")

    hist = m["charts"]["target_hist"]
    centers = [
        (hist["edges"][i] + hist["edges"][i + 1]) / 2
        for i in range(len(hist["counts"]))
    ]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=hist["counts"],
            marker_color=ACCENT,
            hovertemplate="Value %{x:.2f} × $100k<br>%{y} districts<extra></extra>",
        )
    )
    fig.update_layout(
        title="Target distribution (MedHouseVal)",
        xaxis_title="Median house value ($100k)",
        yaxis_title="Block groups",
    )
    st.plotly_chart(dark_layout(fig, 360), use_container_width=True)
    st.caption("The spike at 5.0 is the well-known $500k cap, not a cluster of identical homes.")

    g1, g2 = st.columns(2)
    with g1:
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
        fig.update_layout(
            title="Income versus house value",
            xaxis_title="Median income ($10k)",
            yaxis_title="Median value ($100k)",
        )
        st.plotly_chart(dark_layout(fig, 380), use_container_width=True)
    with g2:
        geo = pd.DataFrame(m["charts"]["geo"])
        fig = go.Figure(
            go.Scatter(
                x=geo["lon"],
                y=geo["lat"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=geo["price"],
                    colorscale="Tealgrn",
                    opacity=0.75,
                    colorbar=dict(title="$100k"),
                ),
                hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{marker.color:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Where expensive lots sit",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
        )
        st.plotly_chart(dark_layout(fig, 380), use_container_width=True)

    st.subheader("Feature dictionary")
    rows = [
        {
            "Feature": col,
            "Label": m["feature_labels"][col],
            "Unit": m["feature_units"][col],
            "Train mean": m["feature_stats"][col]["mean"],
        }
        for col in m["features"]
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format({"Train mean": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    corr = m["charts"]["correlation"]
    heat = go.Figure(
        go.Heatmap(
            z=corr["matrix"],
            x=corr["columns"],
            y=corr["columns"],
            colorscale="Tealgrn",
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr["matrix"]],
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    heat.update_layout(title="Correlation")
    st.plotly_chart(dark_layout(heat, 520), use_container_width=True)

elif page == "Models":
    st.header("Model comparison")
    st.caption("Test set · 4,128 districts · no test-set leakage in preprocessing")

    res = pd.DataFrame(m["models"])
    show = res.rename(
        columns={
            "name": "Model",
            "mae": "MAE",
            "mse": "MSE",
            "rmse": "RMSE",
            "r2": "R²",
            "mae_usd": "MAE ($)",
            "rmse_usd": "RMSE ($)",
        }
    )[["Model", "MAE", "MSE", "RMSE", "R²", "MAE ($)", "RMSE ($)"]]
    st.dataframe(
        show.style.format(
            {
                "MAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R²": "{:.4f}",
                "MAE ($)": "${:,.0f}",
                "RMSE ($)": "${:,.0f}",
            }
        ).highlight_min(subset=["MAE", "MSE", "RMSE", "MAE ($)", "RMSE ($)"], color="#1e4a42")
        .highlight_max(subset=["R²"], color="#1e4a42"),
        use_container_width=True,
        hide_index=True,
    )
    st.success(
        f"Best model: **{winner_name}** (RMSE {money(winner['rmse_usd'])}, R² {winner['r2']:.4f})"
    )

    fig = go.Figure(
        go.Bar(
            x=res["name"],
            y=res["rmse_usd"],
            marker_color=[ACCENT if w else "#c4b5a0" for w in res["winner"]],
            hovertemplate="%{x}<br>RMSE %{y:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Holdout RMSE (lower is better)", yaxis_title="USD")
    st.plotly_chart(dark_layout(fig, 380), use_container_width=True)

    avp = pd.DataFrame(m["charts"]["actual_vs_predicted"])
    resid = pd.DataFrame(m["charts"]["residuals"])
    g1, g2 = st.columns(2)
    with g1:
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
        fig.update_layout(
            title=f"Actual vs predicted · {winner_name}",
            xaxis_title="Actual ($100k)",
            yaxis_title="Predicted ($100k)",
        )
        st.plotly_chart(dark_layout(fig, 420), use_container_width=True)
    with g2:
        fig = go.Figure(
            go.Scatter(
                x=resid["predicted"],
                y=resid["residual"],
                mode="markers",
                marker=dict(size=5, color="#c4b5a0", opacity=0.35),
                hovertemplate="Predicted %{x:.2f}<br>Residual %{y:.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        fig.update_layout(
            title="Residuals vs predicted",
            xaxis_title="Predicted ($100k)",
            yaxis_title="Residual ($100k)",
        )
        st.plotly_chart(dark_layout(fig, 420), use_container_width=True)

    st.markdown(
        f"**Why {winner_name} won.** House prices here mix smooth income gradients "
        "with sharp geographic breaks. A linear plane cannot represent the Bay versus "
        f"the Central Valley with the same slope, which is why RMSE falls from "
        f"{money(linear['rmse_usd'])} (linear) to {money(winner['rmse_usd'])}."
    )

elif page == "Features":
    st.header("Feature importance")
    st.caption(f"From the winning **{winner_name}** model")
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
    fig.update_layout(title="What the booster listens to", xaxis_title="Importance")
    st.plotly_chart(dark_layout(fig, 420), use_container_width=True)

    st.markdown(
        """
        **Reading this**
        - Median income is 37.8% of the signal — ability to pay.
        - Occupancy is the second single feature: crowded block groups trade cheaper.
        - Latitude and longitude together rival occupancy — coastal and Bay Area premiums.
        - Rooms matter more than bedrooms or raw population.
        """
    )

elif page == "Predict":
    st.header("Price a district")
    st.caption(
        "The published champion is XGBoost. Live scoring uses the trained linear "
        "regression coefficients so every slider has an additive, inspectable effect."
    )

    preset_names = ["Average district"] + [p["name"] for p in m["presets"]]
    choice = st.radio("Preset", preset_names, horizontal=True)

    if choice == "Average district":
        base = {col: float(m["feature_stats"][col]["mean"]) for col in m["features"]}
        actual = None
    else:
        found = next(p for p in m["presets"] if p["name"] == choice)
        base = {k: float(v) for k, v in found["features"].items()}
        actual = float(found["actual"])

    features: dict[str, float] = {}
    cols = st.columns(2)
    for i, col in enumerate(m["features"]):
        stat = m["feature_stats"][col]
        with cols[i % 2]:
            features[col] = st.slider(
                f"{m['feature_labels'][col]} ({m['feature_units'][col]})",
                min_value=float(stat["min"]),
                max_value=float(stat["max"]),
                value=float(base[col]),
                step=(float(stat["max"]) - float(stat["min"])) / 200,
            )

    units, parts = predict_linear(m, features)
    usd = max(0.0, units * USD)

    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown(
            f"<div class='metric-card'><p class='kicker'>Linear estimate</p>"
            f"<p style='font-size:2.6rem;margin:0;color:{ACCENT}'>{money(usd)}</p>"
            f"<p class='muted'>{units:.2f} × $100,000 units</p></div>",
            unsafe_allow_html=True,
        )
        if actual is not None:
            st.caption(f"Recorded median for this {choice} sample: {money(actual * USD)}")
    with m2:
        st.markdown("**Feature contributions (standardized × coefficient)**")
        drv = pd.DataFrame(parts, columns=["feature", "contribution"])
        fig = go.Figure(
            go.Bar(
                x=drv["contribution"][::-1],
                y=drv["feature"][::-1],
                orientation="h",
                marker_color=[(ACCENT if v > 0 else "#c4b5a0") for v in drv["contribution"][::-1]],
                hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title="Positive = higher predicted value")
        st.plotly_chart(dark_layout(fig, 360), use_container_width=True)
