Excellent choice 👍
Below I give you **three complete, notebook-ready sections**, written so you can **paste them as cells** and run end-to-end.
All three are **leakage-safe**, **time-series correct**, and **compatible with what you already built**.

I’ll assume you already have:

* `ons_level` → ONS Food & Non-Alcoholic Beverages **level series** (monthly, month-end index)
* `monthly_df` → FG data with `FG_ONLINE`, `FG_OFFLINE`
* A split date like `split = '2024-12-31'`

I will **rebuild features cleanly inside each approach** so there’s no hidden leakage.

---

# 🔵 1️⃣ FULL XGBOOST NOTEBOOK (LEAKAGE-SAFE)

### 📌 Why XGBoost here

* Captures **non-linear lag effects**
* Handles **interactions automatically**
* Works well with **small macro datasets**

---

## 🟦 Cell 1 — Build target (YoY inflation) & FG features

(**YoY works much better for ML than MoM**)

```python
import numpy as np
import pandas as pd

# ---------- TARGET: YoY inflation ----------
ONS_yoy = ons_level.pct_change(12).dropna()
ONS_yoy.name = 'ONS_yoy'

# ---------- FG TOTAL ----------
FG_total = monthly_df['FG_ONLINE'] + monthly_df['FG_OFFLINE']
FG_total = FG_total.loc[ONS_yoy.index]  # align

# ---------- FG FEATURES ----------
features_df = pd.DataFrame(index=ONS_yoy.index)
features_df['FG_lag1'] = FG_total.shift(1)
features_df['FG_lag2'] = FG_total.shift(2)
features_df['FG_lag3'] = FG_total.shift(3)

features_df['FG_roll3'] = FG_total.rolling(3).mean().shift(1)
features_df['FG_roll6'] = FG_total.rolling(6).mean().shift(1)

features_df['FG_vol3'] = FG_total.rolling(3).std().shift(1)

# Online share
features_df['FG_online_share'] = (
    monthly_df['FG_ONLINE'] / (monthly_df['FG_ONLINE'] + monthly_df['FG_OFFLINE'])
).shift(1)

df_ml = pd.concat([ONS_yoy, features_df], axis=1).dropna()
df_ml.head()
```

---

## 🟦 Cell 2 — Train / test split (time-safe)

```python
split = '2024-12-31'

train = df_ml.loc[:split]
test  = df_ml.loc[df_ml.index > pd.to_datetime(split)]

X_train = train.drop(columns='ONS_yoy')
y_train = train['ONS_yoy']

X_test = test.drop(columns='ONS_yoy')
y_test = test['ONS_yoy']
```

---

## 🟦 Cell 3 — Train XGBoost (no leakage)

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred = pd.Series(xgb.predict(X_test), index=y_test.index)
```

---

## 🟦 Cell 4 — Evaluate

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print("XGBoost RMSE:", rmse)
print("XGBoost MAE :", mae)
```

---

## 🟦 Cell 5 — Feature importance (interpretability)

```python
import matplotlib.pyplot as plt

xgb_importance = pd.Series(
    xgb.feature_importances_,
    index=X_train.columns
).sort_values()

xgb_importance.plot(kind='barh', figsize=(8,5))
plt.title("XGBoost Feature Importance")
plt.show()
```

---

# 🟢 2️⃣ ELASTIC NET FEATURE SELECTION (LEAKAGE-SAFE)

### 📌 Why Elastic Net

* Handles **multicollinearity**
* Performs **automatic variable selection**
* Much safer than AIC-only selection

---

## 🟩 Cell 1 — Standardize features (TRAIN ONLY)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_std = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)

X_test_std = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index,
    columns=X_test.columns
)
```

---

## 🟩 Cell 2 — Elastic Net with time-series-safe CV

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

enet = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=np.logspace(-4, 1, 50),
    cv=tscv,
    max_iter=10000
)

enet.fit(X_train_std, y_train)
```

---

## 🟩 Cell 3 — Selected features

```python
enet_coefs = pd.Series(enet.coef_, index=X_train.columns)
selected_features = enet_coefs[enet_coefs != 0]

selected_features.sort_values()
```

👉 These are the **statistically strongest FG drivers**.

---

## 🟩 Cell 4 — Evaluate Elastic Net

```python
y_pred_enet = pd.Series(enet.predict(X_test_std), index=y_test.index)

rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_enet))
mae_enet  = mean_absolute_error(y_test, y_pred_enet)

print("ElasticNet RMSE:", rmse_enet)
print("ElasticNet MAE :", mae_enet)
```

---

# 🔴 3️⃣ SARIMA + ML HYBRID (VERY STRONG)

### 📌 Why this often wins

* SARIMA captures **inflation persistence**
* ML captures **nonlinear demand effects**
* Used in **central bank forecasting**

---

## 🔺 Cell 1 — SARIMA baseline

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima = SARIMAX(
    train['ONS_yoy'],
    order=(1,0,1),
    seasonal_order=(0,0,0,0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_res = sarima.fit(disp=False)

sarima_pred_train = sarima_res.fittedvalues
sarima_pred_test  = sarima_res.get_forecast(len(test)).predicted_mean
sarima_pred_test.index = y_test.index
```

---

## 🔺 Cell 2 — Train ML on SARIMA residuals

```python
residuals_train = y_train - sarima_pred_train

xgb_resid = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

xgb_resid.fit(X_train, residuals_train)
```

---

## 🔺 Cell 3 — Hybrid forecast

```python
resid_pred = pd.Series(
    xgb_resid.predict(X_test),
    index=y_test.index
)

hybrid_pred = sarima_pred_test + resid_pred
```

---

## 🔺 Cell 4 — Hybrid evaluation

```python
rmse_hybrid = np.sqrt(mean_squared_error(y_test, hybrid_pred))
mae_hybrid  = mean_absolute_error(y_test, hybrid_pred)

print("HYBRID RMSE:", rmse_hybrid)
print("HYBRID MAE :", mae_hybrid)
```

---

## 🔺 Cell 5 — Compare all models

```python
comparison = pd.DataFrame({
    'RMSE': [rmse, rmse_enet, rmse_hybrid],
    'MAE' : [mae, mae_enet, mae_hybrid]
}, index=['XGBoost', 'ElasticNet', 'SARIMA+XGB'])

comparison
```

---

# 🧠 HOW TO INTERPRET RESULTS (IMPORTANT)

* **If SARIMA+ML wins** → publish this as your final model
* **If XGBoost ≈ SARIMA** → FG adds weak nonlinear signal
* **If Elastic Net selects few features** → confirms demand effect is small

All outcomes are **valid research conclusions**.

---

# ✅ RECOMMENDATION (HONEST)

If this were my project, I would report:

1. SARIMA baseline
2. SARIMAX (your best version)
3. SARIMA + XGBoost hybrid (final)

That’s **industry-grade macro modelling**.

---

If you want next, I can:

* help you **decide which model to present**
* write **methodology + results text**
* convert this into **a paper-ready table**

Just tell me.
