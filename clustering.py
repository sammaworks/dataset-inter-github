from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class ModelData:

    # ... your existing methods (including apply_sarimax) ...

    @staticmethod
    def _build_lstm_sequences(X_arr, y_arr, idx_arr, lookback):
        """
        Internal helper: create [samples, timesteps, features] and aligned y, index.
        """
        X_seq, y_seq, idx_seq = [], [], []

        for t in range(lookback, len(X_arr)):
            X_seq.append(X_arr[t - lookback:t, :])
            y_seq.append(y_arr[t])
            idx_seq.append(idx_arr[t])

        return np.array(X_seq), np.array(y_seq), np.array(idx_seq)

    @staticmethod
    def apply_lstm(
        train_df,
        test_df,
        benchmark,
        split_point,
        lstm_model_parameters,
        print_model_parameters: bool = False,
        print_comparision_plot: bool = False,
        print_evaluation_matrix: bool = False,
    ):
        """
        LSTM baseline using (benchmark, exogenous features) over a rolling window.

        train_df, test_df : exogenous features (DataFrames with DateTimeIndex)
        benchmark         : target series (pd.Series)
        split_point       : last date of training (Timestamp / str)
        lstm_model_parameters : dict or dataclass.asdict with keys:
            - lookback (int, e.g. 12)
            - epochs (int, e.g. 100)
            - batch_size (int, e.g. 16)
            - units (int, e.g. 32)
            - dropout (float, e.g. 0.2)
            - val_split (float, e.g. 0.2)
            - patience (int, e.g. 10)
            - verbose (0/1/2)
        """

        # -----------------------------
        # 1. Unpack LSTM hyperparams
        # -----------------------------
        lookback = lstm_model_parameters.get("lookback", 12)
        epochs = lstm_model_parameters.get("epochs", 100)
        batch_size = lstm_model_parameters.get("batch_size", 16)
        units = lstm_model_parameters.get("units", 32)
        dropout = lstm_model_parameters.get("dropout", 0.2)
        val_split = lstm_model_parameters.get("val_split", 0.2)
        patience = lstm_model_parameters.get("patience", 10)
        verbose = lstm_model_parameters.get("verbose", 1)

        split_point_ts = pd.to_datetime(split_point)

        # -----------------------------
        # 2. Align full X (train+test) and y
        # -----------------------------
        X_full = pd.concat([train_df, test_df], axis=0).sort_index()
        y_full = benchmark.sort_index()

        common_idx = X_full.index.intersection(y_full.index)
        X_full = X_full.loc[common_idx]
        y_full = y_full.loc[common_idx]

        # Raw arrays
        X_raw = X_full.values.astype("float32")
        y_raw = y_full.values.astype("float32").reshape(-1, 1)
        idx_raw = common_idx.to_numpy()

        # -----------------------------
        # 3. Scale features using TRAIN only
        # -----------------------------
        train_mask_scaler = common_idx <= split_point_ts
        X_train_raw_for_scaler = X_raw[train_mask_scaler]
        y_train_raw_for_scaler = y_raw[train_mask_scaler]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        scaler_X.fit(X_train_raw_for_scaler)
        scaler_y.fit(y_train_raw_for_scaler)

        X_scaled = scaler_X.transform(X_raw)
        y_scaled = scaler_y.transform(y_raw).flatten()  # 1D

        # -----------------------------
        # 4. Build sequences [samples, timesteps, features]
        # -----------------------------
        X_seq, y_seq, idx_seq = ModelData._build_lstm_sequences(
            X_scaled, y_scaled, idx_raw, lookback
        )

        # Now split again into train / test based on prediction time index
        idx_seq_ts = pd.to_datetime(idx_seq)
        train_mask = idx_seq_ts <= split_point_ts

        X_train_seq = X_seq[train_mask]
        y_train_seq = y_seq[train_mask]
        idx_train_seq = idx_seq_ts[train_mask]

        X_test_seq = X_seq[~train_mask]
        y_test_seq = y_seq[~train_mask]
        idx_test_seq = idx_seq_ts[~train_mask]

        print(
            f"LSTM: {X_train_seq.shape[0]} train samples, "
            f"{X_test_seq.shape[0]} test samples, "
            f"lookback={lookback}, features={X_full.shape[1]}"
        )

        # -----------------------------
        # 5. Define LSTM model
        # -----------------------------
        model = Sequential()
        model.add(
            LSTM(
                units,
                input_shape=(lookback, X_full.shape[1]),
                return_sequences=False,
            )
        )
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        if print_model_parameters:
            model.summary()

        es = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose,
        )

        # -----------------------------
        # 6. Predictions (in scaled space)
        # -----------------------------
        y_train_pred_scaled = model.predict(X_train_seq, verbose=0).flatten()
        y_test_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()

        # Inverse transform to original scale
        y_train_pred = scaler_y.inverse_transform(
            y_train_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_test_pred = scaler_y.inverse_transform(
            y_test_pred_scaled.reshape(-1, 1)
        ).flatten()

        y_train_true = scaler_y.inverse_transform(
            y_train_seq.reshape(-1, 1)
        ).flatten()
        y_test_true = scaler_y.inverse_transform(
            y_test_seq.reshape(-1, 1)
        ).flatten()

        y_train_true_s = pd.Series(y_train_true, index=idx_train_seq)
        y_train_pred_s = pd.Series(y_train_pred, index=idx_train_seq)

        y_test_true_s = pd.Series(y_test_true, index=idx_test_seq)
        y_test_pred_s = pd.Series(y_test_pred, index=idx_test_seq)

        # Put into a single df like SARIMAX
        result_train_df = pd.concat([y_train_pred_s, y_train_true_s], axis=1)
        result_test_df = pd.concat([y_test_pred_s, y_test_true_s], axis=1)

        result_df = pd.concat([result_train_df, result_test_df])
        result_df.columns = ["Predicted", "Benchmark"]
        result_df = result_df.loc[TIMEPOINTS.final_results_start_year:]

        # -----------------------------
        # 7. Optional comparison plot
        # -----------------------------
        if print_comparision_plot:
            fig = result_df.plot(title="LSTM – Predicted vs Benchmark")
            fig.add_vline(
                x=split_point_ts,
                line_width=2,
                line_color="red",
                line_dash="dash",
            )
            fig.show()

        # -----------------------------
        # 8. Optional evaluation metrics
        # -----------------------------
        if print_evaluation_matrix:
            # Align to compute metrics
            benchmark_train = result_df["Benchmark"].loc[:split_point_ts]
            benchmark_test = result_df["Benchmark"].loc[split_point_ts:]

            predicted_train = result_df["Predicted"].loc[benchmark_train.index]
            predicted_test = result_df["Predicted"].loc[benchmark_test.index]

            mse_train = mean_squared_error(benchmark_train, predicted_train)
            mse_test = mean_squared_error(benchmark_test, predicted_test)
            rmse_train = sqrt(mse_train)
            rmse_test = sqrt(mse_test)

            print(
                "Training vs Testing MSE:",
                mse_train,
                ">>",
                mse_test,
            )
            print(
                "Training vs Testing RMSE:",
                rmse_train,
                ">>",
                rmse_test,
            )

            print(
                "Pearson Correlation\n",
                result_df.corr(),
            )

        return result_df
