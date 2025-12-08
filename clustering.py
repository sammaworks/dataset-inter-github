class ModelData:
    ...
    @staticmethod
    def apply_stlforecast(
        train_df,
        test_df,
        benchmark,
        split_point,
        period=12,
        arima_order=(1, 0, 0),
        print_model_summary=False,
        print_comparision_plot=True,
        print_evaluation_matrix=True,
    ):
        """
        STL + ARIMA forecaster with exogenous variables.

        - Fits STL + ARIMA only on the TRAIN period (<= split_point)
        - STL handles seasonality (period), ARIMA handles deseasonalized series
        - Seasonality for future is generated using the final STL seasonal cycle
          (statsmodels STLForecast default behaviour – no leakage).
        - Exogenous regressors (train_df / test_df) are used in the ARIMA part.

        Args
        ----
        train_df : pd.DataFrame or pd.Series
            Exogenous variables for training period.
        test_df : pd.DataFrame or pd.Series
            Exogenous variables for testing period.
        benchmark : pd.Series
            Target series (e.g. your benchmark_input).
        split_point : str or pd.Timestamp
            Date separating train and test.
        period : int
            Seasonal period for STL (12 for monthly yearly seasonality).
        arima_order : tuple
            (p, d, q) order for the ARIMA model on the deseasonalized series.
        print_model_summary : bool
            If True, prints underlying ARIMA summary.
        print_comparision_plot : bool
            If True, plots Predicted vs Benchmark with vertical split line.
        print_evaluation_matrix : bool
            If True, prints MSE / RMSE and Pearson/Spearman for
            overall, train, and test.

        Returns
        -------
        result_df : pd.DataFrame
            DataFrame with 'Predicted' and 'Benchmark' over full period.
        """

        # ------------------------------
        # 0) Basic cleanup & alignment
        # ------------------------------
        split_point = pd.to_datetime(split_point)

        # Ensure Series → DataFrame for exogs
        if isinstance(train_df, pd.Series):
            train_df = train_df.to_frame()
        if isinstance(test_df, pd.Series):
            test_df = test_df.to_frame()

        # Sort everything by index just in case
        benchmark = benchmark.sort_index()
        train_df = train_df.sort_index()
        test_df = test_df.sort_index()

        # First split benchmark by date
        y_train = benchmark.loc[:split_point]
        y_test = benchmark.loc[split_point:]

        # Align exogenous data to benchmark indices
        X_train = train_df.loc[y_train.index.intersection(train_df.index)]
        y_train = y_train.loc[X_train.index]

        X_test = test_df.loc[y_test.index.intersection(test_df.index)]
        y_test = y_test.loc[X_test.index]

        # Sanity check
        # print(len(X_train), len(y_train), len(X_test), len(y_test))

        # --------------------------------
        # 1) STLForecast (no leakage)
        # --------------------------------
        # STL is fitted only on y_train (and X_train), not on future data.
        stlf = STLForecast(
            endog=y_train,
            exog=X_train,
            period=period,
            model=ARIMA,
            model_kwargs={"order": arima_order},
        )

        stlf_res = stlf.fit()

        if print_model_summary:
            print(stlf_res.model_result.summary())

        # In-sample fitted values for train
        # (predict uses same STL + ARIMA, no future info)
        predicted_train = stlf_res.predict(
            start=y_train.index[0],
            end=y_train.index[-1],
            exog=X_train,
        )

        # Out-of-sample forecast for test horizon
        forecast_test = stlf_res.forecast(
            steps=len(y_test),
            exog=X_test,
        )

        # --------------------------------
        # 2) Combine results
        # --------------------------------
        result_train_df = pd.concat(
            [predicted_train.rename("Predicted"), y_train.rename("Benchmark")],
            axis=1,
        )
        result_test_df = pd.concat(
            [forecast_test.rename("Predicted"), y_test.rename("Benchmark")],
            axis=1,
        )

        result_df = pd.concat([result_train_df, result_test_df], axis=0)

        # --------------------------------
        # 3) Plot comparison
        # --------------------------------
        if print_comparision_plot:
            fig = result_df.plot()
            fig.axvline(
                x=split_point,
                linewidth=2,
                color="red",
                linestyle="dashed",
            )

        # --------------------------------
        # 4) Evaluation metrics
        # --------------------------------
        if print_evaluation_matrix:
            # MSE / RMSE
            mse_train = mean_squared_error(
                y_train, predicted_train.loc[y_train.index]
            )
            rmse_train = sqrt(mse_train)

            mse_test = mean_squared_error(
                y_test, forecast_test.loc[y_test.index]
            )
            rmse_test = sqrt(mse_test)

            print("Training vs Testing MSE:", mse_train, " >> ", mse_test)
            print("Training vs Testing RMSE:", rmse_train, " >> ", rmse_test)

            # Overall correlations
            valid_idx = result_df[["Predicted", "Benchmark"]].dropna().index
            pred_all = result_df.loc[valid_idx, "Predicted"]
            bench_all = result_df.loc[valid_idx, "Benchmark"]

            print("Pearson Correlation (Overall):\n", pred_all.corr(bench_all))
            if len(pred_all) > 1:
                spear_overall, _ = spearmanr(pred_all, bench_all)
                print("Spearman Correlation (Overall):\n", spear_overall)

            # Train correlations
            train_idx = result_train_df.dropna().index
            pred_tr = result_train_df.loc[train_idx, "Predicted"]
            bench_tr = result_train_df.loc[train_idx, "Benchmark"]
            print("Pearson Correlation (Train):\n", pred_tr.corr(bench_tr))
            if len(pred_tr) > 1:
                spear_tr, _ = spearmanr(pred_tr, bench_tr)
                print("Spearman Correlation (Train):\n", spear_tr)

            # Test correlations
            test_idx = result_test_df.dropna().index
            pred_te = result_test_df.loc[test_idx, "Predicted"]
            bench_te = result_test_df.loc[test_idx, "Benchmark"]
            print("Pearson Correlation (Test):\n", pred_te.corr(bench_te))
            if len(pred_te) > 1:
                spear_te, _ = spearmanr(pred_te, bench_te)
                print("Spearman Correlation (Test):\n", spear_te)

        return result_df
