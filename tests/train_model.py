import time
from pickle import dump

import joblib
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
from darts.models import LinearRegressionModel, NHiTSModel
from darts_config import *
from darts_functions import count_zeroes, get_relevent_columns, preprocess_data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Customize model arguements
custom_pl_trainer_kwargs = pl_trainer_kwargs.copy()
custom_pl_trainer_kwargs["max_epochs"] = 200
custom_pl_trainer_kwargs["callbacks"] = [EarlyStopping(**early_stopping_args)]


custom_args = common_model_args.copy()
custom_args["output_chunk_length"] = FORECAST_PERIODS
custom_args["pl_trainer_kwargs"] = custom_pl_trainer_kwargs

SKU_INDEX = 0

if __name__ == "__main__":
    # Track time
    start_time = time.time()

    # Load data
    file_path = "tests/resources/Apr24_telefire.xlsx"
    raw_df = pd.read_excel(file_path)

    # Get the index of the forecast date
    FORECAST_DATE = raw_df["Forecast Date"][0]
    df_sliced = get_relevent_columns(data=raw_df, forecast_date=FORECAST_DATE)

    # Preprocess data frame
    df = preprocess_data(raw_data=raw_df, data_df=df_sliced, column_index="Item Code")

    # Sanity check for date range and values
    print(f"Actual sales dates range from {df.index[0]} to {df.index[-1]}")
    print(f"Number of Zeros: {count_zeroes(df)}\nNumber of NaNs: {(df.isna().sum().sum())}")

    # Initalize time series objects for darts
    all_series = [TimeSeries.from_dataframe(df, value_cols=col) for col in df.columns]

    # NHits Neural Network
    scaler = Scaler()
    train_series_list = [scaler.fit_transform(series) for series in all_series]

    print(f"Training set includes {len(df)} months and {len(df.columns)} items")
    nhits = NHiTSModel(**custom_args)
    nhits.fit(train_series_list)

    # Backtesting
    series_scaled = train_series_list[SKU_INDEX]
    TRAIN_WINDOW = len(df) - VAL_WINDOW  # set train window for backtesting
    backtest = nhits.historical_forecasts(
        series_scaled,
        start=TRAIN_WINDOW,
        forecast_horizon=FORECAST_PERIODS,
        stride=1,
        retrain=False,
        verbose=False,
    )

    # Average RMSE
    scores = {}
    for index, forecast in enumerate(backtest):
        forecast = scaler.inverse_transform(forecast)
        series = train_series_list[index]
        rmse_value = rmse(series, forecast)
        scores[series.components[0]] = rmse_value

    mean_scores = sum(scores.values()) / len(scores)
    print("Average RMSE:", mean_scores)

    # Save scaler and model
    nhits.save("nhits_model.pkl")
    dump(scaler, open("scaler.pkl", "wb"))

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
