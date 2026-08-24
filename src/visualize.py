"""Plot helpers. Each function saves a PNG and returns the matplotlib figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import TARGET

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams |= {
    "figure.facecolor": "#F3EEE4",
    "axes.facecolor": "#FCFAF6",
    "axes.edgecolor": "#1C1915",
    "axes.labelcolor": "#1C1915",
    "text.color": "#1C1915",
    "xtick.color": "#1C1915",
    "ytick.color": "#1C1915",
    "grid.color": "#1C191515",
    "font.size": 11,
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


def plot_target_distribution(y: pd.Series, dest: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(y, bins=40, kde=True, ax=ax, color="#3E5363")
    ax.set_xlabel("Median house value ($100k)")
    ax.set_ylabel("Block groups")
    ax.set_title("Target distribution")
    return _save(fig, dest)


def plot_correlation_heatmap(frame: pd.DataFrame, dest: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = frame.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", ax=ax, square=True)
    ax.set_title("Feature correlations")
    return _save(fig, dest)


def plot_income_vs_price(frame: pd.DataFrame, dest: Path, sample: int = 2500) -> Path:
    subset = frame.sample(n=min(sample, len(frame)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(subset["MedInc"], subset[TARGET], s=8, alpha=0.35, c="#3E5363")
    ax.set_xlabel("Median income (tens of thousands USD)")
    ax.set_ylabel("Median house value ($100k)")
    ax.set_title("Income versus house value")
    return _save(fig, dest)


def plot_geography(frame: pd.DataFrame, dest: Path, sample: int = 4000) -> Path:
    subset = frame.sample(n=min(sample, len(frame)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 7))
    pts = ax.scatter(
        subset["Longitude"],
        subset["Latitude"],
        c=subset[TARGET],
        s=8,
        cmap="mako",
        alpha=0.7,
    )
    fig.colorbar(pts, ax=ax, label="Median house value ($100k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California block groups")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, dest)


def plot_model_comparison(table: pd.DataFrame, dest: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=table, x="Model", y="RMSE", ax=axes[0], color="#3E5363")
    sns.barplot(data=table, x="Model", y="R2", ax=axes[1], color="#1C1915")
    axes[0].set_title("Test RMSE (lower is better)")
    axes[1].set_title("Test R² (higher is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
    return _save(fig, dest)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, dest: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.3, c="#3E5363")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#1C1915", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Actual ($100k)")
    ax.set_ylabel("Predicted ($100k)")
    ax.set_title("Actual versus predicted")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, dest)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, dest: Path) -> Path:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(residuals, bins=40, kde=True, ax=axes[0], color="#3E5363")
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Actual − predicted ($100k)")
    axes[1].scatter(y_pred, residuals, s=10, alpha=0.3, c="#3E5363")
    axes[1].axhline(0, color="#1C1915", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Predicted ($100k)")
    axes[1].set_ylabel("Residual ($100k)")
    axes[1].set_title("Residuals versus predicted")
    return _save(fig, dest)


def plot_feature_importance(importance: pd.DataFrame, dest: Path) -> Path:
    ordered = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["feature"], ordered["importance"], color="#3E5363")
    ax.set_xlabel("Relative importance")
    ax.set_title("Feature importance")
    return _save(fig, dest)
