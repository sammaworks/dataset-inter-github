import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from math import sqrt


def apply_sarimax_regularized(
        train_df,
        test_df,
        benchmark,
        split_point,
        order_grid,
        seasonal_order_grid,
        use_bic=False,
        print_model_parameters=False,
        print_comparision_plot=False,
        print_evaluation_matrix=False
):
    """
    Regularized SARIMAX for time series with many exogenous variables.

    Idea:
      1) Use RidgeCV on the exogenous variables (X) to compress them into a
         single regularized index x_reg(t) = f(X_t).
      2) Fit SARIMAX on y_t with this 1-dim exogenous regressor.
      3) Choose SARIMA orders via AIC or BIC (IC = likelihood – penalty*k).

    Args
    ----
    train_df : pd.DataFrame
        Exogenous variables for the training period (before split_point).
    test_df : pd.DataFrame
        Exogenous variables for the testing period (starting at split_point).
    benchmark : pd.Series
        Target series (full period; same index as train_df+test_df).
    split_point : str or pd.Timestamp
        Date index where train ends and test begins (inclusive for test).
    order_grid : list of tuple
        List of (p,d,q) tuples to try, e.g. [(0,1,1), (1,1,1), (2,1,1)].
    seasonal_order_grid : list of tuple
        List of seasonal (P,D,Q,s) tuples to try, e.g. [(0,0,0,0), (0,1,1,12)].
    use_bic : bool
        If True, select model by BIC; else by AIC.
    print_model_parameters : bool
        If True, print summary of the final SARIMAX model.
    print_comparision_plot : bool
        If True, plot Predicted vs Benchmark with a vertical split line.
    print_evaluation_matrix : bool
        If True, print RMSE and correlations for train/test/overall.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame indexed by time with columns:
        ['Predicted', 'Benchmark'] for the full period.
    """

    # ------------------------------------------------------------------
    # 1) Split target into train/test and ensure DataFrame shapes
    # ------------------------------------------------------------------
    benchmark_train = benchmark.loc[:split_point]
    benchmark_test = benchmark.loc[split_point:]

    if isinstance(train_df, pd.Series):
        train_df = train_df.to_frame()
    if isinstance(test_df, pd.Series):
        test_df = test_df.to_frame()

    # Align indices (important!)
    train_df = train_df.loc[benchmark_train.index]
    test_df = test_df.loc[benchmark_test.index]

    # ------------------------------------------------------------------
    # 2) Regularize exogenous variables with RidgeCV
    # ------------------------------------------------------------------
    # Ridge shrinks coefficients and combats overfitting from many exogs.
    alphas = np.logspace(-4, 4, 20)  # wide range of penalty strengths
    ridge = RidgeCV(alphas=alphas, cv=5)

    ridge.fit(train_df, benchmark_train)

    # Build a single regularized exogenous signal for SARIMAX
    exog_train_reg = pd.Series(
        ridge.predict(train_df),
        index=train_df.index,
        name="x_reg"
    )
    exog_test_reg = pd.Series(
        ridge.predict(test_df),
        index=test_df.index,
        name="x_reg"
    )

    exog_train = exog_train_reg.to_frame()
    exog_test = exog_test_reg.to_frame()

    if print_model_parameters:
        print("\nRidge regularization on exogenous features")
        print("Chosen alpha (L2 strength):", ridge.alpha_)
        print("Number of original exogs :", train_df.shape[1])
        print("Now passing 1-dim exog to SARIMAX: 'x_reg'")

    # ------------------------------------------------------------------
    # 3) Grid search SARIMA orders via IC (AIC or BIC)
    # ------------------------------------------------------------------
    best_ic = np.inf
    best_order = None
    best_seasonal_order = None
    best_res = None

    for order in order_grid:
        for seasonal_order in seasonal_order_grid:
            try:
                model = SARIMAX(
                    benchmark_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                res = model.fit(disp=False)

                ic = res.bic if use_bic else res.aic

                if ic < best_ic:
                    best_ic = ic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_res = res

            except Exception as e:
                # Some configurations may fail to converge; skip them
                continue

    if best_res is None:
        raise RuntimeError("No SARIMAX specification converged. "
                           "Check order_grid/seasonal_order_grid.")

    if print_model_parameters:
        print("\nBest SARIMAX orders selected by {}:".format(
            "BIC" if use_bic else "AIC"))
        print("  order         =", best_order)
        print("  seasonal_order=", best_seasonal_order)
        print("  {} value      = {:.3f}".format(
            "BIC" if use_bic else "AIC", best_ic))
        print("\nSARIMAX summary (regularized exog):")
        print(best_res.summary())

    # ------------------------------------------------------------------
    # 4) In-sample prediction (train) & out-of-sample forecast (test)
    # ------------------------------------------------------------------
    # In-sample fitted values for train set
    predicted_train = best_res.predict(
        start=benchmark_train.index[0],
        end=benchmark_train.index[-1],
        exog=exog_train
    )

    # Forecast for test set
    forecast_test = best_res.predict(
        start=benchmark_test.index[0],
        end=benchmark_test.index[-1],
        exog=exog_test
    )

    # Build result DataFrame for full period
    result_train_df = pd.concat(
        [predicted_train.rename("Predicted"),
         benchmark_train.rename("Benchmark")],
        axis=1
    )
    result_test_df = pd.concat(
        [forecast_test.rename("Predicted"),
         benchmark_test.rename("Benchmark")],
        axis=1
    )

    result_df = pd.concat([result_train_df, result_test_df])
    result_df = result_df.sort_index()

    # ------------------------------------------------------------------
    # 5) Plot comparison
    # ------------------------------------------------------------------
    if print_comparision_plot:
        fig = result_df.plot()
        fig.add_vline(
            x=split_point,
            line_width=2,
            line_color="red",
            line_dash="dash"
        )
        fig.show()

    # ------------------------------------------------------------------
    # 6) Evaluation metrics
    # ------------------------------------------------------------------
    if print_evaluation_matrix:
        # Train metrics
        mse_train = mean_squared_error(benchmark_train, predicted_train)
        rmse_train = sqrt(mse_train)

        # Test metrics
        mse_test = mean_squared_error(benchmark_test, forecast_test)
        rmse_test = sqrt(mse_test)

        print("\nTraining vs Testing MSE:", mse_train, " >> ", mse_test)
        print("Training vs Testing RMSE:", rmse_train, " >> ", rmse_test)

        # Overall correlations
        valid_all = result_df.dropna()
        pearson_overall = valid_all["Predicted"].corr(valid_all["Benchmark"])
        spearman_overall, _ = spearmanr(
            valid_all["Predicted"], valid_all["Benchmark"]
        )
        print("\nPearson Correlation (Overall):")
        print(pearson_overall)
        print("Spearman Correlation (Overall):")
        print(spearman_overall)

        # Train correlations
        train_corr_df = result_df.loc[:split_point].dropna()
        pearson_train = train_corr_df["Predicted"].corr(
            train_corr_df["Benchmark"]
        )
        spearman_train, _ = spearmanr(
            train_corr_df["Predicted"], train_corr_df["Benchmark"]
        )
        print("\nPearson Correlation (Train):")
        print(pearson_train)
        print("Spearman Correlation (Train):")
        print(spearman_train)

        # Test correlations
        test_corr_df = result_df.loc[result_df.index > split_point].dropna()
        pearson_test = test_corr_df["Predicted"].corr(
            test_corr_df["Benchmark"]
        )
        spearman_test, _ = spearmanr(
            test_corr_df["Predicted"], test_corr_df["Benchmark"]
        )
        print("\nPearson Correlation (Test):")
        print(pearson_test)
        print("Spearman Correlation (Test):")
        print(spearman_test)

    return result_df
