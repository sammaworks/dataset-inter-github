Below are **full notebook-style changes, cell-by-cell**, for BOTH scenarios — written so you can paste into your notebook with minimum disruption and **no data leakage**.

I’m assuming you already have these objects from your notebook:

* `EconInd_df['ONS Food and nonalcoholic beverages']` (level index series)
* `monthly_df` with `FG_ONLINE`, `FG_OFFLINE` (levels) indexed by month
* `combined_df` already built (or we rebuild cleanly below)
* `split` date string (you used 2024-12-31 style)

---

# ✅ Scenario 1 — Winsorization + Standardization (Leakage-safe)

## Cell A — Build %change series + align dates (replace your ONS/FG pctchange creation + concat)

```python
import numpy as np
import pandas as pd

# ---------- ONS (level -> month-end -> pct change) ----------
ons_level = EconInd_df['ONS Food and nonalcoholic beverages'].copy()
ons_level.index = pd.to_datetime(ons_level.index)
ons_level.index = ons_level.index.to_period("M").to_timestamp("M")
ons_level = ons_level.sort_index()

ONS_FNB_pct = ons_level.pct_change().dropna()
ONS_FNB_pct.name = "ONS_FNB_pct"

# ---------- FG (monthly_df already exists; ensure month-end) ----------
monthly_df = monthly_df.copy()
monthly_df.index = pd.to_datetime(monthly_df.index)
monthly_df.index = monthly_df.index.to_period("M").to_timestamp("M")
monthly_df = monthly_df.sort_index()

FG1_pctchange_df = monthly_df[['FG_ONLINE', 'FG_OFFLINE']].pct_change().dropna()
FG1_pctchange_df = FG1_pctchange_df.rename(columns={
    'FG_ONLINE': 'FG1_ONLINE_pctchange',
    'FG_OFFLINE': 'FG1_OFFLINE_pctchange'
})

# ---------- Combine ----------
combined_df = pd.concat([ONS_FNB_pct, FG1_pctchange_df], axis=1).dropna()
combined_df.head()
```

---

## Cell B — Create lag/rolling exogs (no leakage) + drop NA (replace your lag creation)

```python
full_df = combined_df.copy()

# Lags
for lag in [1, 2]:
    full_df[f'FG1_ONLINE_pctchange_lag{lag}']  = full_df['FG1_ONLINE_pctchange'].shift(lag)
    full_df[f'FG1_OFFLINE_pctchange_lag{lag}'] = full_df['FG1_OFFLINE_pctchange'].shift(lag)

# Rolling mean (strictly past information)
full_df['FG1_ONLINE_pctchange_roll3']  = full_df['FG1_ONLINE_pctchange'].rolling(3).mean().shift(1)
full_df['FG1_OFFLINE_pctchange_roll3'] = full_df['FG1_OFFLINE_pctchange'].rolling(3).mean().shift(1)

full_df = full_df.dropna()
full_df.head()
```

---

## Cell C — Split (fix split duplication bug) (replace your split cell)

```python
split = '2024-12-31'  # month-end

train_df = full_df.loc[:split].copy()
test_df  = full_df.loc[full_df.index > pd.to_datetime(split)].copy()

train_df.shape, test_df.shape
```

---

## Cell D — Winsorize exogs using TRAIN ONLY (NEW cell)

```python
def winsorize_fit(series, lower_q=0.01, upper_q=0.99):
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return lo, hi

def winsorize_apply(series, lo, hi):
    return series.clip(lower=lo, upper=hi)

exog_cols = [
    'FG1_ONLINE_pctchange_lag1',
    'FG1_ONLINE_pctchange_lag2',
    'FG1_OFFLINE_pctchange_lag1',
    'FG1_OFFLINE_pctchange_lag2',
    'FG1_ONLINE_pctchange_roll3',
    'FG1_OFFLINE_pctchange_roll3'
]

train_exog = train_df[exog_cols].copy()
test_exog  = test_df[exog_cols].copy()

winsor_limits = {}
for col in exog_cols:
    lo, hi = winsorize_fit(train_exog[col], 0.01, 0.99)  # tune 0.02/0.98 if needed
    winsor_limits[col] = (lo, hi)

    train_exog[col] = winsorize_apply(train_exog[col], lo, hi)
    test_exog[col]  = winsorize_apply(test_exog[col],  lo, hi)

train_exog.describe(), test_exog.describe()
```

---

## Cell E — Standardize exogs using TRAIN ONLY (NEW cell)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(train_exog),
    index=train_exog.index,
    columns=exog_cols
)

X_test = pd.DataFrame(
    scaler.transform(test_exog),
    index=test_exog.index,
    columns=exog_cols
)

X_train.head()
```

---

## Cell F — Fit SARIMAX + forecast (replace your fit/forecast cell)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

y_train = train_df['ONS_FNB_pct']
y_test  = test_df['ONS_FNB_pct']

sarimax_model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_results = sarimax_model.fit(disp=False)
print(sarimax_results.summary())

# Forecast test
forecast_res = sarimax_results.get_forecast(steps=len(y_test), exog=X_test)
y_test_pred = forecast_res.predicted_mean
y_test_pred.index = y_test.index

# In-sample predictions (train)
y_train_pred = sarimax_results.predict(
    start=y_train.index[0],
    end=y_train.index[-1],
    exog=X_train
)
y_train_pred.index = y_train.index
```

---

## Cell G — Metrics (fix MAPE to avoid inf) (replace your metric cell)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def MAPE_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Test
mae_test  = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mape_test = MAPE_safe(y_test, y_test_pred)

# Train
mae_train  = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mape_train = MAPE_safe(y_train, y_train_pred)

print("MAE_test :", mae_test)
print("RMSE_test:", rmse_test)
print("MAPE_test:", mape_test, "%")

print("MAE_train :", mae_train)
print("RMSE_train:", rmse_train)
print("MAPE_train:", mape_train, "%")
```

---

# ✅ Scenario 2 — Using log diffs (recommended for stability)

Here we switch from `%change` to **log-differences**:

[
\Delta \log(x_t) = \log(x_t) - \log(x_{t-1})
]

This is often better behaved and less sensitive to outliers.

## Cell A2 — Build LOG-DIFF series + align dates (replace pct_change cell)

```python
import numpy as np
import pandas as pd

# ---------- ONS (level -> month-end -> log-diff) ----------
ons_level = EconInd_df['ONS Food and nonalcoholic beverages'].copy()
ons_level.index = pd.to_datetime(ons_level.index)
ons_level.index = ons_level.index.to_period("M").to_timestamp("M")
ons_level = ons_level.sort_index()

ONS_FNB_ld = np.log(ons_level).diff().dropna()
ONS_FNB_ld.name = "ONS_FNB_ld"

# ---------- FG (monthly_df -> month-end -> log-diff) ----------
monthly_df = monthly_df.copy()
monthly_df.index = pd.to_datetime(monthly_df.index)
monthly_df.index = monthly_df.index.to_period("M").to_timestamp("M")
monthly_df = monthly_df.sort_index()

FG_ld_df = np.log(monthly_df[['FG_ONLINE', 'FG_OFFLINE']]).diff().dropna()
FG_ld_df = FG_ld_df.rename(columns={
    'FG_ONLINE': 'FG1_ONLINE_ld',
    'FG_OFFLINE': 'FG1_OFFLINE_ld'
})

combined_df = pd.concat([ONS_FNB_ld, FG_ld_df], axis=1).dropna()
combined_df.head()
```

---

## Cell B2 — Create lag/rolling features for log-diffs (replace lag cell)

```python
full_df = combined_df.copy()

for lag in [1, 2]:
    full_df[f'FG1_ONLINE_ld_lag{lag}']  = full_df['FG1_ONLINE_ld'].shift(lag)
    full_df[f'FG1_OFFLINE_ld_lag{lag}'] = full_df['FG1_OFFLINE_ld'].shift(lag)

full_df['FG1_ONLINE_ld_roll3']  = full_df['FG1_ONLINE_ld'].rolling(3).mean().shift(1)
full_df['FG1_OFFLINE_ld_roll3'] = full_df['FG1_OFFLINE_ld'].rolling(3).mean().shift(1)

full_df = full_df.dropna()
full_df.head()
```

---

## Cell C2 — Split (same)

```python
split = '2024-12-31'

train_df = full_df.loc[:split].copy()
test_df  = full_df.loc[full_df.index > pd.to_datetime(split)].copy()

train_df.shape, test_df.shape
```

---

## Cell D2 — (Optional but recommended) Winsorize + Standardize log-diff exogs (train-only)

Log-diffs are usually less spiky, but you can still apply the same exog prep.

```python
from sklearn.preprocessing import StandardScaler

exog_cols = [
    'FG1_ONLINE_ld_lag1',
    'FG1_ONLINE_ld_lag2',
    'FG1_OFFLINE_ld_lag1',
    'FG1_OFFLINE_ld_lag2',
    'FG1_ONLINE_ld_roll3',
    'FG1_OFFLINE_ld_roll3'
]

train_exog = train_df[exog_cols].copy()
test_exog  = test_df[exog_cols].copy()

# Winsorize on train only
winsor_limits = {}
for col in exog_cols:
    lo = train_exog[col].quantile(0.01)
    hi = train_exog[col].quantile(0.99)
    winsor_limits[col] = (lo, hi)
    train_exog[col] = train_exog[col].clip(lo, hi)
    test_exog[col]  = test_exog[col].clip(lo, hi)

# Standardize on train only
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(train_exog), index=train_exog.index, columns=exog_cols)
X_test  = pd.DataFrame(scaler.transform(test_exog), index=test_exog.index, columns=exog_cols)
```

---

## Cell E2 — Fit + forecast (log-diff target)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_train = train_df['ONS_FNB_ld']
y_test  = test_df['ONS_FNB_ld']

model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

res = model.fit(disp=False)
print(res.summary())

fc = res.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
fc.index = y_test.index

pred_train = res.predict(start=y_train.index[0], end=y_train.index[-1], exog=X_train)
pred_train.index = y_train.index

# Metrics (MAE/RMSE; MAPE not meaningful for log-diff)
mae_test  = mean_absolute_error(y_test, fc)
rmse_test = np.sqrt(mean_squared_error(y_test, fc))

mae_train  = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))

print("MAE_test :", mae_test)
print("RMSE_test:", rmse_test)
print("MAE_train:", mae_train)
print("RMSE_train:", rmse_train)
```

> Note: MAPE is usually **not appropriate** for log-diff because the target is not a “level” ratio. Use RMSE/MAE.

---

# Which scenario should you choose?

* **Winsor + Standardize**: best when FG % changes are very spiky and hurting fit.
* **Log diffs**: best when you want a cleaner statistical series and more stable coefficients.

If you want, I can also add a final cell that runs **both scenarios side-by-side** and prints a comparison table (SARIMA vs SARIMAX, RMSE/MAE/MAPE/SMAPE).
