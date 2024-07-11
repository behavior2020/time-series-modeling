import sys

import joblib
import pandas as pd
from darts import TimeSeries
from darts.models import NHiTSModel
from darts_config import *
from darts_functions import count_zeroes, get_relevent_columns, preprocess_data

if __name__ == "__main__":
    # Load model
    model = NHiTSModel.load("nhits_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Check correct number of arguments
    if len(sys.argv) != 2:
        print("Wrong number of arugments")
        sys.exit(1)

    # Get path to new data
    new_data = sys.argv[1]  # 2nd argument because 1st is always the script
    raw_df = pd.read_excel(new_data)

    # Get the index of the forecast date
    FORECAST_DATE = raw_df["Forecast Date"][0]
    df_sliced = get_relevent_columns(data=raw_df, forecast_date=FORECAST_DATE)

    # Preprocess data frame
    df = preprocess_data(raw_data=raw_df, data_df=df_sliced, column_index="Item Code")

    # Sanity check for date range and values
    print(f"Actual sales dates range from {df.index[0]} to {df.index[-1]}")
    print(f"Number of Zeros: {count_zeroes(df)}\nNumber of NaNs: {(df.isna().sum().sum())}")

    # Initalize time series objects for darts
    new_series = [TimeSeries.from_dataframe(df, value_cols=col) for col in df.columns]
    TRAIN_WINDOW = len(df) - VAL_WINDOW  # set train window for backtesting

    new_series_scaled = [scaler.transform(series) for series in new_series]  # only transform on new data

    # Train-Test split
    SKU_INDEX = 0

    # Predict
    predictions = model.predict(n=12, series=new_series_scaled[SKU_INDEX])
    predictions = scaler.inverse_transform(predictions)

    print(predictions)
