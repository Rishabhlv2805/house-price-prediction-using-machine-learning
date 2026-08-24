"""Train the four regressors and score them on the same holdout fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .data import RANDOM_STATE, SplitData

USD_PER_UNIT = 100_000  # MedHouseVal is stored in $100k units


@dataclass(frozen=True)
class ModelScore:
    name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    estimator: RegressorMixin
    uses_scaled_features: bool
    y_pred: np.ndarray


def _decision_tree() -> GridSearchCV:
    """Light depth search — a single deep tree overfits this dataset quickly."""
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid={
            "max_depth": [6, 8, 10, 12],
            "min_samples_leaf": [4, 8, 16],
        },
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    return grid


def build_estimators() -> dict[str, tuple[RegressorMixin, bool]]:
    """Map display name → (estimator, needs_scaled_X).

    Linear regression is scale-sensitive, so it sees standardized features.
    Axis-aligned trees ignore monotonic transforms, so they stay on the
    winsorized but unscaled matrices.
    """
    return {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (_decision_tree(), False),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }


def evaluate_estimator(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_and_score(split: SplitData) -> list[ModelScore]:
    """Train every model on the same split and score the held-out test fold."""
    y_test = split.y_test.to_numpy()
    results: list[ModelScore] = []
    for name, (estimator, scaled) in build_estimators().items():
        X_train = split.X_train_scaled if scaled else split.X_train
        X_test = split.X_test_scaled if scaled else split.X_test
        estimator.fit(X_train, split.y_train)
        fitted = estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator
        y_pred = fitted.predict(X_test)
        metrics = evaluate_estimator(y_test, y_pred)
        results.append(
            ModelScore(
                name=name,
                mae=metrics["mae"],
                mse=metrics["mse"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                estimator=fitted,
                uses_scaled_features=scaled,
                y_pred=np.asarray(y_pred, dtype=float),
            )
        )
    results.sort(key=lambda row: row.rmse)
    return results


def scores_to_frame(scores: list[ModelScore]) -> pd.DataFrame:
    rows = [
        {
            "Model": score.name,
            "MAE": score.mae,
            "MSE": score.mse,
            "RMSE": score.rmse,
            "R2": score.r2,
        }
        for score in scores
    ]
    return pd.DataFrame(rows)


def feature_importance_frame(estimator: RegressorMixin, columns: list[str]) -> pd.DataFrame:
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(f"{type(estimator).__name__} has no feature_importances_")
    frame = pd.DataFrame(
        {"feature": columns, "importance": estimator.feature_importances_}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
