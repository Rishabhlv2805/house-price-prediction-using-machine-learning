"""Load, inspect, and split the California Housing dataset.

The target (MedHouseVal) is median house value in units of $100,000.
Features are already numeric, so encoding is unnecessary. The pipeline
still audits missing values and winsorizes a handful of physically
implausible occupancy / room ratios that are known data-entry artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "MedHouseVal"
# Extreme AveOccup / AveRooms values (e.g. 1,000+ occupants) are almost
# certainly encoding errors in a handful of block groups.
OUTLIER_COLUMNS = ("AveRooms", "AveBedrms", "Population", "AveOccup")
WINSOR_LIMITS = (0.01, 0.99)
FEATURE_COLUMNS = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


@dataclass(frozen=True)
class SplitData:
    """Train/test matrices plus the scaler fitted on train only."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler


def load_housing_frame() -> pd.DataFrame:
    """Return the sklearn California Housing frame with a stable column order."""
    bundle = fetch_california_housing(as_frame=True)
    frame = bundle.frame.copy()
    return frame[list(FEATURE_COLUMNS) + [TARGET]]


def missing_value_report(frame: pd.DataFrame) -> pd.Series:
    """Count nulls per column. California Housing is complete; this proves it."""
    return frame.isna().sum()


def winsorize_outliers(
    frame: pd.DataFrame,
    columns: Iterable[str] = OUTLIER_COLUMNS,
    limits: tuple[float, float] = WINSOR_LIMITS,
) -> pd.DataFrame:
    """Clip selected columns to empirical quantiles computed on this frame.

    Call this on the full exploratory copy for plots. For modeling, quantiles
    are estimated on the *training* split only (see `preprocess`) so the test
    fold cannot influence the clip thresholds.
    """
    cleaned = frame.copy()
    lower_q, upper_q = limits
    for column in columns:
        low, high = cleaned[column].quantile([lower_q, upper_q])
        cleaned[column] = cleaned[column].clip(lower=low, upper=high)
    return cleaned


def _clip_with_bounds(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    clipped = frame.copy()
    for column, (low, high) in bounds.items():
        clipped[column] = clipped[column].clip(lower=low, upper=high)
    return clipped


def preprocess(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> SplitData:
    """Split 80/20, winsorize using train quantiles, scale using train stats.

    Tree models consume the unscaled matrices; linear regression consumes
    the scaled copies. The scaler is fitted on train only.
    """
    X = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    bounds: dict[str, tuple[float, float]] = {}
    for column in OUTLIER_COLUMNS:
        low, high = X_train[column].quantile(list(WINSOR_LIMITS))
        bounds[column] = (float(low), float(high))

    X_train = _clip_with_bounds(X_train, bounds)
    X_test = _clip_with_bounds(X_test, bounds)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
    )


def train_test_matrices(frame: pd.DataFrame) -> SplitData:
    """Alias kept for a shorter import in the notebook."""
    return preprocess(frame)
