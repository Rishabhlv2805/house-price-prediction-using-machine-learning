"""CLI: train all models, write reports, persist the winner, export the web payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `python -m src.train` from the ml/ directory.
if __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLUMNS, TARGET, load_housing_frame, missing_value_report, preprocess
from src.models import (
    USD_PER_UNIT,
    feature_importance_frame,
    fit_and_score,
    scores_to_frame,
)
from src.visualize import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_geography,
    plot_income_vs_price,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _sample_pairs(y_true: np.ndarray, y_pred: np.ndarray, n: int = 900, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    return [
        {"actual": float(y_true[i]), "predicted": float(y_pred[i])}
        for i in idx
    ]


def _histogram(values: np.ndarray, bins: int = 36) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
    }


def _region(lat: float, lon: float) -> str:
    if lat >= 37.7 and lon <= -121.7:
        return "Bay Area"
    if lat >= 36.6 and lon <= -121.2:
        return "Central Coast"
    if lat >= 38.8:
        return "Northern California"
    if lat <= 33.0:
        return "San Diego"
    if lat <= 34.4 and lon <= -118.05:
        return "Los Angeles"
    if lon >= -117.6:
        return "Inland Empire"
    if lon >= -119.2 and lat <= 36.2:
        return "Central Valley"
    return "Interior California"


def _export_xgb_booster(estimator) -> dict | None:
    if not hasattr(estimator, "get_booster"):
        return None
    booster = estimator.get_booster()
    trees = [json.loads(blob) for blob in booster.get_dump(dump_format="json")]
    config = json.loads(booster.save_config())
    learner = config.get("learner", {})
    param = learner.get("learner_model_param", {})
    raw_base = param.get("base_score", 0.5)
    if isinstance(raw_base, str):
        try:
            parsed = json.loads(raw_base)
            raw_base = parsed[0] if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            raw_base = float(raw_base)
    if isinstance(raw_base, list):
        raw_base = raw_base[0]
    return {"base_score": float(raw_base), "trees": trees}


def export_web_payload(
    frame: pd.DataFrame,
    split,
    scores,
    importance: pd.DataFrame,
    winner,
) -> dict:
    y_test = split.y_test.to_numpy()
    winner_pred = winner.y_pred
    corr = frame.corr(numeric_only=True)

    scaler = split.scaler
    linear = next(s for s in scores if s.name == "Linear Regression")
    lr = linear.estimator

    rng = np.random.default_rng(42)
    geo_idx = rng.choice(len(frame), size=min(2200, len(frame)), replace=False)
    geo = frame.iloc[geo_idx]
    income_idx = rng.choice(len(frame), size=min(1200, len(frame)), replace=False)
    income = frame.iloc[income_idx]

    feature_stats = {}
    for col in FEATURE_COLUMNS:
        series = split.X_train[col]
        feature_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "p05": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
        }

    presets = []
    labeled = frame.copy()
    labeled["_region"] = [_region(r.Latitude, r.Longitude) for r in labeled.itertuples()]
    for region in (
        "Bay Area",
        "Los Angeles",
        "San Diego",
        "Central Valley",
        "Central Coast",
        "Inland Empire",
    ):
        subset = labeled[labeled["_region"] == region]
        if subset.empty:
            continue
        row = subset.sample(n=1, random_state=7).iloc[0]
        presets.append(
            {
                "name": region,
                "features": {col: float(row[col]) for col in FEATURE_COLUMNS},
                "actual": float(row[TARGET]),
            }
        )

    # Trees are several MB; the web studio uses the linear explainer instead.
    xgb_bundle = None
    xgb_score = next((s for s in scores if s.name == "XGBoost"), None)

    verify = []
    sample_idx = rng.choice(len(split.X_test), size=12, replace=False)
    if xgb_score is not None:
        for i in sample_idx:
            feats = {col: float(split.X_test.iloc[i][col]) for col in FEATURE_COLUMNS}
            verify.append(
                {
                    "features": feats,
                    "xgb": float(xgb_score.y_pred[i]),
                    "linear": float(linear.y_pred[i]),
                    "actual": float(y_test[i]),
                }
            )

    describe = frame.describe().round(4).to_dict()

    payload = {
        "dataset": {
            "name": "California Housing",
            "source": "sklearn.datasets.fetch_california_housing",
            "n_samples": int(len(frame)),
            "n_features": len(FEATURE_COLUMNS),
            "target": TARGET,
            "target_unit": "100000 USD",
            "usd_per_unit": USD_PER_UNIT,
            "missing_values": {k: int(v) for k, v in missing_value_report(frame).items()},
            "describe": describe,
            "notes": [
                "MedHouseVal is censored at 5.00001 ($500,000), which compresses the upper tail.",
                "A handful of AveOccup / AveRooms values are physically implausible and were winsorized at the 1st/99th training percentiles.",
            ],
        },
        "features": list(FEATURE_COLUMNS),
        "feature_labels": {
            "MedInc": "Median income",
            "HouseAge": "Median house age",
            "AveRooms": "Average rooms",
            "AveBedrms": "Average bedrooms",
            "Population": "Block population",
            "AveOccup": "Average occupancy",
            "Latitude": "Latitude",
            "Longitude": "Longitude",
        },
        "feature_units": {
            "MedInc": "tens of thousands USD",
            "HouseAge": "years",
            "AveRooms": "rooms / household",
            "AveBedrms": "bedrooms / household",
            "Population": "people",
            "AveOccup": "people / household",
            "Latitude": "degrees",
            "Longitude": "degrees",
        },
        "feature_stats": feature_stats,
        "models": [
            {
                "name": s.name,
                "mae": s.mae,
                "mse": s.mse,
                "rmse": s.rmse,
                "r2": s.r2,
                "mae_usd": s.mae * USD_PER_UNIT,
                "rmse_usd": s.rmse * USD_PER_UNIT,
                "winner": s.name == winner.name,
            }
            for s in scores
        ],
        "winner": winner.name,
        "importance": importance.to_dict(orient="records"),
        "charts": {
            "target_hist": _histogram(frame[TARGET].to_numpy()),
            "correlation": {
                "columns": list(corr.columns),
                "matrix": corr.round(4).values.tolist(),
            },
            "income": [
                {"x": float(r.MedInc), "y": float(r.MedHouseVal)}
                for r in income.itertuples()
            ],
            "geo": [
                {
                    "lon": float(r.Longitude),
                    "lat": float(r.Latitude),
                    "price": float(r.MedHouseVal),
                }
                for r in geo.itertuples()
            ],
            "actual_vs_predicted": _sample_pairs(y_test, winner_pred),
            "residuals": [
                {
                    "predicted": p["predicted"],
                    "residual": p["actual"] - p["predicted"],
                }
                for p in _sample_pairs(y_test, winner_pred, n=900, seed=7)
            ],
        },
        "linear": {
            "intercept": float(lr.intercept_),
            "coefficients": {
                col: float(coef) for col, coef in zip(FEATURE_COLUMNS, lr.coef_)
            },
            "scaler_mean": {col: float(m) for col, m in zip(FEATURE_COLUMNS, scaler.mean_)},
            "scaler_scale": {col: float(m) for col, m in zip(FEATURE_COLUMNS, scaler.scale_)},
        },
        "xgboost": xgb_bundle,
        "verify": verify,
        "presets": presets,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train house-price models.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--web-json",
        action="append",
        type=Path,
        default=[],
        help="Optional extra JSON payload path (studio export).",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading California Housing…")
    frame = load_housing_frame()
    print(f"  shape={frame.shape}  missing={int(frame.isna().sum().sum())}")

    split = preprocess(frame)
    print(f"  train={len(split.X_train)}  test={len(split.X_test)}")

    print("Training four models (this takes a minute)…")
    scores = fit_and_score(split)
    table = scores_to_frame(scores)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    winner = scores[0]
    print(f"\nWinner: {winner.name}  RMSE={winner.rmse:.4f}  R²={winner.r2:.4f}")

    importance = feature_importance_frame(winner.estimator, list(FEATURE_COLUMNS))
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    joblib.dump(winner.estimator, ARTIFACTS / "best_model.joblib")
    joblib.dump(split.scaler, ARTIFACTS / "preprocessor.joblib")

    if not args.skip_plots:
        print("Writing plots…")
        plot_target_distribution(frame[TARGET], ARTIFACTS / "01_target_distribution.png")
        plot_correlation_heatmap(frame, ARTIFACTS / "02_correlation_heatmap.png")
        plot_income_vs_price(frame, ARTIFACTS / "03_income_vs_price.png")
        plot_geography(frame, ARTIFACTS / "04_geography.png")
        plot_model_comparison(table, ARTIFACTS / "05_model_comparison.png")
        plot_actual_vs_predicted(
            split.y_test.to_numpy(), winner.y_pred, ARTIFACTS / "06_actual_vs_predicted.png"
        )
        plot_residuals(split.y_test.to_numpy(), winner.y_pred, ARTIFACTS / "07_residuals.png")
        plot_feature_importance(importance, ARTIFACTS / "08_feature_importance.png")

    if args.web_json:
        payload = export_web_payload(frame, split, scores, importance, winner)
        for dest in args.web_json:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload))
            print(f"Wrote {dest} ({dest.stat().st_size // 1024} KB)")

    print("Done.")


if __name__ == "__main__":
    main()
