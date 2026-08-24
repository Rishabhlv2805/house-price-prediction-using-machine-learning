# House Price Prediction using Machine Learning

Author: **Rishabh Sharma** ([Rishabhlv2805](https://github.com/Rishabhlv2805))

**Live app:** [https://house-price-prediction-using-machine-learning-rs.streamlit.app/](https://house-price-prediction-using-machine-learning-rs.streamlit.app/)

Predict median house value for California census block groups using the public California Housing dataset. Four models are trained on the same preprocessed split and compared on MAE, MSE, RMSE, and R².

Interactive dashboard: `streamlit_app.py` (Overview, Dataset, Models, Features, live Predict).

The Python pipeline in `notebooks/` and `src/` is the reproducible training path.

## Overview

House prices mix income, occupancy, and location. This project scores median block-group value from those signals so the models can be compared fairly on a held-out test set.

## Dataset

- **Name:** California Housing
- **Source:** [scikit-learn — fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Size:** 20,640 districts × 8 features + target
- **Target:** `MedHouseVal` — median house value in units of $100,000, censored at 5.00001 ($500,000)
- **Note:** The frame is fully numeric and has 0 missing values. Extreme `AveOccup` / `AveRooms` ratios are treated as data-entry artefacts and winsorized at the 1st/99th **training** percentiles so the test fold cannot leak.

| Column | Meaning | Unit |
|--------|---------|------|
| MedInc | Median block-group income | tens of thousands USD |
| HouseAge | Median house age | years |
| AveRooms | Average rooms per household | count |
| AveBedrms | Average bedrooms per household | count |
| Population | Block-group population | people |
| AveOccup | Average household occupancy | people |
| Latitude / Longitude | Block-group centroid | degrees |
| MedHouseVal (target) | Median house value | $100,000s |

## Project structure

```
├── streamlit_app.py
├── notebooks/01_house_price_prediction.ipynb
├── src/data.py
├── src/models.py
├── src/visualize.py
├── src/train.py
├── artifacts/
├── reports/model_comparison.csv
├── reports/metrics.json
├── screenshots/
├── requirements.txt
├── requirements-train.txt
└── README.md
```

## Models used

- Linear Regression
- Decision Tree (grid over `max_depth` ∈ {6, 8, 10, 12} and `min_samples_leaf` ∈ {4, 8, 16})
- Random Forest (`n_estimators=200`)
- XGBoost (`n_estimators=300`, `learning_rate=0.05`, `max_depth=6`)

**Split:** 80/20, shuffled, `random_state=42`.

**Preprocessing:** winsorize occupancy/room outliers using train quantiles; `StandardScaler` fitted on train only for linear regression. Tree models use the unscaled matrix.

## Results

Test set (4,128 districts). Target units are $100,000; dollar figures multiply by 100,000.

| Model               | MAE              | MSE    | RMSE             | R²     |
|---------------------|------------------|--------|------------------|--------|
| Linear Regression   | 0.4981 ($49.8k)  | 0.4620 | 0.6797 ($68.0k)  | 0.6474 |
| Decision Tree       | 0.4045 ($40.5k)  | 0.3575 | 0.5979 ($59.8k)  | 0.7272 |
| Random Forest       | 0.3269 ($32.7k)  | 0.2551 | 0.5050 ($50.5k)  | 0.8054 |
| XGBoost             | **0.2992 ($29.9k)** | **0.2049** | **0.4526 ($45.3k)** | **0.8437** |

**Best model:** XGBoost (lowest RMSE, highest R²). Saved to `artifacts/best_model.joblib`.

## Streamlit dashboard

Five pages, same layout as the churn project:

| Page | What it shows |
|------|----------------|
| Overview | Sample size, best model, R², project summary |
| Dataset | Target distribution, income vs price, California map, correlation heatmap |
| Models | MAE / MSE / RMSE / R² table, RMSE bars, actual vs predicted, residuals |
| Features | XGBoost importance (income 37.8%, occupancy, lat/long) |
| Predict | Sliders + Bay Area / Inland / Crowded presets; live linear estimate in USD |

Live scoring uses the trained linear coefficients so every slider has an additive, inspectable effect. The published champion remains XGBoost.

## Key insights

- Median income is 37.8% of XGBoost importance — ability to pay is the loudest price signal.
- Latitude + longitude together are ~26% — coastal and Bay Area premiums remain after income is controlled.
- Average occupancy is the second single feature (13.8%) — crowded block groups trade cheaper.
- Rooms matter more than bedrooms or raw population.
- The $500k label cap shows up in the residuals. Errors fan out as predictions approach 5.0 because the labels themselves are clipped.

## How to run the Streamlit app

**Live:** [https://house-price-prediction-using-machine-learning-rs.streamlit.app/](https://house-price-prediction-using-machine-learning-rs.streamlit.app/)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How to run the notebook

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
jupyter notebook notebooks/01_house_price_prediction.ipynb
```

## How to retrain

```bash
pip install -r requirements-train.txt
python -m src.train
```

Writes `reports/model_comparison.csv`, plots under `artifacts/`, and `artifacts/best_model.joblib`.

## Technologies

pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib, jupyter, streamlit, plotly.

`requirements.txt` is the Streamlit Cloud install (pandas, numpy, streamlit, plotly). Use `requirements-train.txt` to retrain.
