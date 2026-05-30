# IronKaggle 🏆 — Predictive Sales Modeling

A machine learning project built as part of the IronHack Data Analytics bootcamp. The goal is to predict daily store sales using historical retail data.

## Project Structure

```
├── sales.csv                     # Training data
├── ironkaggle_notarget.csv       # Test data (no target variable)
├── ironkaggle_student.ipynb      # Main notebook
└── ironkaggle_predictions.csv    # Final predictions
```

## Dataset

**Training data:** `sales.csv` — 640,840 rows of daily store records.

| Column | Description |
|---|---|
| `Store_ID` | Unique store identifier |
| `Day_of_week` | Day encoded 1–7 |
| `Date` | Date of the record |
| `Nb_customers_on_day` | Number of customers that day |
| `Open` | 1 = open, 0 = closed |
| `Promotion` | 1 = promotion active, 0 = none |
| `State_holiday` | 0 = none, a/b/c = different holiday types |
| `School_holiday` | 1 = school holiday, 0 = none |
| `Sales` | 🎯 Target variable — total daily sales revenue |

## Approach

### Feature Engineering
- Extracted date components: year, month, day, week of year
- Created flags: `is_weekend`, `is_month_start`, `is_month_end`
- Encoded `State_holiday` as ordinal (0/1/2/3)
- Interaction features: `customers × open`, `customers × promo`, `open × promo`

### Models Trained
- Linear Regression (baseline)
- Ridge Regression
- Bagging
- Random Forest
- Gradient Boosting
- **XGBoost** ← best performer

### Tuning
`GridSearchCV` with 3-fold CV on XGBoost, tuning `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.

## Results

XGBoost (tuned) came out on top across all metrics. Key insight: **foot traffic (`nb_customers`) is the strongest predictor of sales** — the interaction feature `customers_x_open` ranked #1 in feature importance.

One hard rule applied at prediction time: if `Open = 0`, sales are forced to 0 (100% consistent in the training data).

## How to Run

1. Make sure you have the dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

2. Place `sales.csv` and `ironkaggle_notarget.csv` in the same folder as the notebook.

3. Run all cells in `ironkaggle_student.ipynb` — predictions will be saved to `ironkaggle_predictions.csv`.

## What I'd Improve With More Time
- Store-level aggregate features (each store has its own sales profile)
- Lag features — previous day/week sales are likely very predictive
- SHAP values for better model interpretability
- Tune Gradient Boosting as well — it was competitive without any tuning
