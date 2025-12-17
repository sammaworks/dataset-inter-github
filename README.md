Great question — this is an **important modelling concept**, and it often makes a **bigger difference than standardizing** when your exogenous variables are noisy.

I’ll explain it **intuitively**, then **statistically**, then show **exact code** (leakage-safe).

---

## 1️⃣ What does “spiky exogs” mean in your case?

Your FG %-change series looks like this (from your plots):

* Many small month-to-month changes
* **Occasional very large jumps** (±20%, ±30%, even more)
* These jumps are often:

  * promotions
  * reporting artefacts
  * calendar effects
  * one-off shocks

These are **outliers**, not the “typical” demand signal.

---

## 2️⃣ Why standardizing does NOT fix this

### What standardization does

Standardization (z-score):

[
x_{\text{std}} = \frac{x - \mu}{\sigma}
]

This:

* rescales values
* centers them at 0
* **does NOT remove outliers**

⚠️ In fact:

* extreme values remain extreme
* they still dominate likelihood estimation
* they can distort ARMA + exog coefficient estimation

So after standardization:

* a +40% FG spike is still a **huge leverage point**
* SARIMAX will try to “explain” that spike
* forecasts degrade

---

## 3️⃣ What winsorization does (key idea)

**Winsorization caps extreme values** instead of rescaling everything.

Example (1%–99% winsorization):

* Any value below 1st percentile → set to 1st percentile
* Any value above 99th percentile → set to 99th percentile
* All other values remain unchanged

So:

* extreme spikes are **tamed**
* normal variation is preserved
* time structure is not broken

This is **much better for macro + retail series**.

---

## 4️⃣ Why winsorization works better for SARIMAX

SARIMAX assumes:

* roughly Gaussian residuals
* linear response to exogs

Extreme FG spikes:

* violate Gaussian assumptions
* cause coefficient instability
* inflate AR terms
* reduce forecast accuracy

Winsorization:

* improves residual normality
* stabilizes coefficients
* improves out-of-sample RMSE more than scaling

That’s why many applied macro/energy/retail papers do this.

---

## 5️⃣ VERY IMPORTANT: avoid data leakage

You must:

* compute winsorization thresholds **on training data only**
* apply the same caps to test data

Never compute percentiles on the full dataset.

---

## 6️⃣ Correct, leakage-safe winsorization code (use this)

### Step 1: define a winsorization function

```python
def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi), lo, hi
```

---

### Step 2: fit caps on TRAIN only

```python
exog_cols = [
    'FG1_ONLINE_pctchange_lag2',
    'FG1_OFFLINE_pctchange_lag2',
    'FG1_ONLINE_pctchange_roll3',
    'FG1_OFFLINE_pctchange_roll3'
]

train_exog = train_df[exog_cols].copy()
test_exog  = test_df[exog_cols].copy()

winsor_limits = {}

for col in exog_cols:
    train_exog[col], lo, hi = winsorize_series(train_exog[col])
    winsor_limits[col] = (lo, hi)
```

---

### Step 3: apply SAME caps to test data

```python
for col, (lo, hi) in winsor_limits.items():
    test_exog[col] = test_exog[col].clip(lower=lo, upper=hi)
```

Now:

* train and test are treated consistently
* no future information leaked

---

## 7️⃣ How winsorization changes your model behavior

After winsorization, you’ll usually see:

* smaller but **more stable coefficients**
* fewer covariance warnings
* AR coefficient slightly lower (less forced persistence)
* better **test RMSE / MAE**
* cleaner residual diagnostics

Even if AIC worsens slightly, **forecast accuracy often improves**.

---

## 8️⃣ Should you still standardize after winsorizing?

You can, but it’s optional.

Best practice order:

1. **Winsorize**
2. (Optional) Standardize
3. Fit SARIMAX

If you standardize **without winsorizing**, outliers still dominate.

---

## 9️⃣ When winsorization is NOT appropriate

Do NOT winsorize if:

* spikes are the phenomenon you want to model (e.g., crisis detection)
* you’re doing causal inference on extreme shocks

But for **forecasting inflation using retail demand** → winsorization is appropriate.

---

## 10️⃣ Simple intuition summary (one sentence)

> Standardization rescales spikes; winsorization removes their leverage — and leverage is what breaks SARIMAX.

---

If you want, next I can:

* add winsorization + standardization cleanly into your existing notebook
* show **before vs after** RMSE comparison
* help you decide optimal percentile (1–99 vs 2–98)

Just tell me 👍
