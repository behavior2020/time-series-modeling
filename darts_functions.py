import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse
from darts.models import (
    ExponentialSmoothing,
    NaiveDrift,
    NaiveEnsembleModel,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    NHiTSModel,
    Prophet,
    RegressionEnsembleModel,
    StatsForecastAutoCES,
    StatsForecastAutoETS,
    StatsForecastAutoTheta,
    TiDEModel,
)
from darts_config import *
from darts_config import ZEROS_MAX_PERCENT


# Preprocessing & data loading functions
def get_relevent_columns(data: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """Slices data to include only the actual sales data"""
    end_col = data.columns.get_loc(forecast_date)

    # Get the index of the first date column
    start_col = 0
    for index, value in enumerate(data.columns):
        try:
            pd.to_datetime(value, format="%d/%m/%Y")
            start_col = index
            break
        except ValueError:
            pass

    df = data.copy()
    df = df.iloc[:, start_col:end_col]  # includes only relevant dates
    return df


def preprocess_data(raw_data: pd.DataFrame, data_df: pd.DataFrame, column_index: str) -> pd.DataFrame:
    """Returns preprocessed data set for modeling"""
    data_df.columns = pd.to_datetime(
        data_df.columns, format="%d%m%Y"
    )  # convert date columns to datetime and transpose dataframe
    data_df.set_index(raw_data[column_index], inplace=True)
    data_df = data_df.T  # transpose dataframe
    data_df.columns = data_df.columns.astype(str)  # convert item names to string
    # data_df.dropna(axis=1, inplace=True)  # drop all columns with at least one NaN value
    data_df.fillna(0, inplace=True)  # fill missing values with 0
    data_df.iloc[:, 1:] = data_df.iloc[:, 1:].clip(lower=0)  # clip negative values to zero
    data_df = data_df.loc[
        :, (data_df == 0).sum(axis=0) <= (ZEROS_MAX_PERCENT * data_df.shape[0])
    ]  # drop columns with more than n percent of zeros
    # data_df.replace(0, np.nan, inplace=True)  # replace zeros with NaN
    # data_df.interpolate(method="linear", axis=1, inplace=True)  # interpolate missing values column wise
    return data_df


def preprocess_sparse_data(raw_data: pd.DataFrame, data_df: pd.DataFrame, column_index: str) -> pd.DataFrame:
    """Returns preprocessed data set for modeling"""
    data_df.columns = pd.to_datetime(
        data_df.columns, format="%d%m%Y"
    )  # convert date columns to datetime and transpose dataframe
    data_df.set_index(raw_data[column_index], inplace=True)
    data_df = data_df.T  # transpose dataframe
    data_df.columns = data_df.columns.astype(str)  # convert item names to string
    # data_df.dropna(axis=1, inplace=True)  # drop all columns with at least one NaN value
    data_df.iloc[:, 1:] = data_df.iloc[:, 1:].clip(lower=0)  # clip negative values to zero
    data_df = data_df.loc[
        :, (data_df == 0).sum(axis=0) >= (ZEROS_MAX_PERCENT * data_df.shape[0])
    ]  # drop columns with more than n percent of zeros
    # data_df.replace(0, np.nan, inplace=True)  # replace zeros with NaN
    # data_df.interpolate(method="linear", axis=1, inplace=True)  # interpolate missing values column wise
    data_df.fillna(0, inplace=True)  # forward fill missing values
    return data_df


def initialize_models() -> tuple[list[tuple], list[tuple]]:
    # Establishes all models for testing
    stat_models = [
        (StatsForecastAutoETS, {"season_length": SEASONALITY}),
        (StatsForecastAutoCES, {"season_length": SEASONALITY}),
        (StatsForecastAutoTheta, {"season_length": SEASONALITY}),
        (ExponentialSmoothing, {"seasonal_periods": SEASONALITY}),
        (NaiveSeasonal, {"K": SEASONALITY}),
        (NaiveDrift, {}),
        (NaiveMovingAverage, {"input_chunk_length": SEASONALITY}),
        (NaiveMean, {}),
        (
            Prophet,
            {
                "add_seasonality": {
                    "name": "monthly",
                    "seasonal_periods": DAYS_PER_MONTH,
                    "fourier_order": FOURIER_ORDER,
                }
            },
        ),
    ]

    # Add extra parameter to TiDEModel
    tide_model_args = common_model_args.copy()
    tide_model_args.update(
        {
            "use_reversible_instance_norm": True,
        }
    )
    # Choose deep learning models for testing
    deep_learning_models = [
        (
            NHiTSModel,
            common_model_args,
        ),  # DARTS configues the model and training parameters (number of layers,nodes, and type of layers (convolutional for NHiTS))
        (TiDEModel, tide_model_args),
    ]

    return stat_models, deep_learning_models


def initialize_df() -> tuple[str, pd.DataFrame]:
    """Starts the data extraction and preprocessing steps"""
    # Load data
    file_path = file_name
    raw_df = pd.read_excel(file_path)

    # Get the index of the forecast date
    FORECAST_DATE = raw_df["Forecast Date"][0]
    df_sliced = get_relevent_columns(data=raw_df, forecast_date=FORECAST_DATE)

    # Preprocess data frame
    df = preprocess_data(raw_data=raw_df, data_df=df_sliced, column_index="Item Code")

    # Sanity check for date range and values
    print(f"Actual sales dates range from {df.index[0]} to {df.index[-1]}")
    print(f"Number of Zeros: {count_zeroes(df)}\nNumber of NaNs: {(df.isna().sum().sum())}")

    return file_path, df


# Smaller functions
def calc_nMAPE_series(actual: pd.Series, forecast: pd.Series) -> float:
    """Calculates normalized Mean Absolute Percentage Error (nMAPE)"""
    df = pd.DataFrame({"actual": actual, "forecast": forecast})

    # Calculate the absolute differences between actual and forecast
    diff = np.abs(df["actual"] - df["forecast"])

    # Calculate the maximum values between actual and forecast for each row
    max_val = df[["actual", "forecast"]].max(axis=1)

    # Calculate the nMAPE score
    score = round(np.mean((diff / max_val).replace([np.inf], 0)), 2)

    return score


def calc_nMAPE(actual: TimeSeries, forecast: TimeSeries) -> float:
    """Calculates normalized Mean Absolute Percentage Error (nMAPE) for Darts TimeSeries"""

    aligned_actual = actual.slice_intersect(forecast)  # Align the actual and forecast TimeSeries
    actual_values = aligned_actual.pd_dataframe().squeeze()  # Convert TimeSeries to pandas Series
    forecast_values = forecast.pd_dataframe().squeeze()
    df = pd.DataFrame({"Actual": actual_values, "Forecast": forecast_values})
    diff = np.abs(df["Actual"] - df["Forecast"])
    max_val = df[["Actual", "Forecast"]].max(axis=1)  # Take the maximum value between Actual and Forecast
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle divisions by zero
        mape = diff / max_val
        mape = mape.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
    return round(np.nanmean(mape) * 100, 2)  # Return nMAPE as a percentage


def custom_backtest(
    model, series: TimeSeries, train_window: int, forecast_hoizon: int, retrain: bool = True, verbose: bool = True
) -> list:
    """Runs darts backtest with custom parameters"""

    backtest = model.historical_forecasts(
        series,
        start=train_window,
        forecast_horizon=forecast_hoizon,
        stride=1,
        retrain=retrain,
        verbose=verbose,
    )
    return backtest


def plot_models(model, forecast: TimeSeries, baseline_forecast: TimeSeries, series: TimeSeries, label: str):
    """Plot backtest against baseline predictions with RMSE"""

    print(f"Backtest includes {len(forecast)} predictions")
    # Plot backtest model
    series.plot(label=series.components[0])
    forecast.plot(label=label, lw=2)
    baseline_forecast.plot(label="baseline")

    plt.title(type(model).__name__)
    rmse_value = round(rmse(series, forecast), 2)
    plt.title(f"{type(model).__name__}\nRMSE = {rmse_value}")
    plt.show()


def create_csv(df: pd.DataFrame, file_path: str, name: str) -> None:
    """Creates a csv file with the best model for each item based on average NMAPE"""
    df.to_csv(f"{file_path.split('/')[-1][:-5]}_darts_results_{name}.csv", index=False)


def count_zeroes(df: pd.DataFrame) -> pd.Series:
    """Counts the number of zeroes in the dataframe"""
    return df.isin([0]).sum().sum()


def calc_cov(data: pd.Series) -> float:
    """Calculate coefficient of variation (CoV)"""
    mean = np.mean(data)
    std = np.std(data)
    return round(std / mean, 2)


def clean_input_df(raw_df: pd.DataFrame) -> None:
    forecast_date = raw_df["Forecast Date"][0]
    end_col = raw_df.columns.get_loc(forecast_date)
    item_col_str = raw_df.columns.get_loc("Item Code") or raw_df.columns.get_loc("item_code")

    # Get the index of the first date column
    start_col = 0
    for index, value in enumerate(raw_df.columns):
        try:
            pd.to_datetime(value, format="%d/%m/%Y")
            start_col = index
            break
        except ValueError:
            pass

    df = raw_df.copy()
    df = df.iloc[:, start_col:end_col]


def count_results(data: pd.DataFrame) -> dict:
    """Returns the number of times a model won"""
    results = {}
    for index, row in data.iterrows():
        item = row[0]
        subset = data[data["Item"] == item]
        optimizer = Optimizer(subset)
        model = optimizer.get_best_model(column_name="RMSE")
        results[item] = model

    model_counts = Counter(results.values())
    return model_counts


# Get most optimal results from csv
class Optimizer:
    def __init__(self, df: pd.DataFrame):
        """Initializes the optimizer with the given dataframe"""
        self.df = df

    def get_best_model(self, column_name: str):
        """Returns the model name with the lowest value for the given column"""
        self.model_name = self.df["Model Type"][self.df[column_name].idxmin()]
        self.column_name = column_name
        return self.model_name

    def get_best_metric(self, column_name: str):
        """Returns the lowest value for the given column"""
        self.metric = self.df[column_name].min()
        return self.metric

    def print_summary(self):
        """Prints all the results in the dataframe"""
        print(f"{self.column_name}: {self.metric}, {self.model_name}")


# Full summary functions
def calc_statistical_results(
    model_list: list[tuple[object, dict]],
    data: pd.DataFrame,
    series_list: list[TimeSeries],
    file_path: str,
    train_percentage: float,
    prediction_length: int,
) -> pd.DataFrame:
    """Returns a dataframe with model results for each item"""

    results = []
    model_instances = {}
    for model_class, params in model_list:
        try:
            # Log the process start for each item and model
            print(f"Processing model {model_class.__name__}...")
            start_time = time.time()  # start timer

            # Handle model-specifc initialization
            if model_class == Prophet:
                model = model_class()
                model.add_seasonality(**params["add_seasonality"])
            else:
                model = model_class(**params)

            # Perform historical forecasting
            backtest_results = model.historical_forecasts(
                series_list,
                start=train_percentage,
                forecast_horizon=prediction_length,
                stride=1,
                retrain=True,
                verbose=True,
            )

            for i, backtest in enumerate(backtest_results):
                current_nmape = calc_nMAPE(series_list[i], backtest)  # calculates normalized MAPE
                current_rmse = rmse(series_list[i], backtest)  # calculates RMSE
                current_mae = mae(series_list[i], backtest)  # calculates MAE
                file_name = file_path.split("/")[-1][:-5]  # get the name of the fill
                cv = calc_cov(data.iloc[:, i])  # calculates coefficient of variation

                item_results = {
                    "Item": data.columns[i],
                    "Client": file_name,
                    "nMAPE": round(current_nmape, 2),
                    "RMSE": round(current_rmse, 2),
                    "MAE": round(current_mae, 2),
                    "Model Type": type(model).__name__,
                    "Time Taken": round(time.time() - start_time, 3),
                    "CoV": cv,
                }

                results.append(item_results)
            model_instances[model_class.__name__] = model  # store the fitted model for later use
        except Exception as e:
            # Log the error with details
            print(f"Error processing model {model_class.__name__}: {e}")

    print("Processing complete!")
    return pd.DataFrame(results), model_instances


def calc_nn_results(
    model_list: list[tuple[object, dict]],
    data: pd.DataFrame,
    series_list: list[TimeSeries],
    file_path: str,
    train_percentage: float,
) -> pd.DataFrame:
    """Returns a dataframe with model results for each item"""

    results = []
    model_instances = {}
    scaler = Scaler()

    for model_class, params in model_list:
        try:
            # Log the process start for each item and model
            print(f"Processing model {model_class.__name__}...")
            start_time = time.time()  # start timer
            model = model_class(**params)
            series_list_scaled = scaler.fit_transform(series_list)
            model.fit(series_list_scaled)  # fits all items at once

            # Perform historical forecasting
            backtest_results = model.historical_forecasts(
                series_list_scaled,
                start=train_percentage,  # if too high, backtest cannot run enough validation sets.
                forecast_horizon=params["output_chunk_length"],
                stride=1,
                retrain=False,  # "True" means the model is being "retrained" or "fitted" again at every step of the historical forecast
                verbose=True,
                show_warnings=False,
            )

            prediction = scaler.inverse_transform(backtest_results)

            for i, series in enumerate(series_list):
                current_nmape = calc_nMAPE(series, prediction[i])  # calculates normalized MAPE
                current_rmse = rmse(series, prediction[i])  # calculates MAPE
                current_mae = mae(series, prediction[i])  # calculates MAE
                file_name = file_path.split("/")[-1][:-5]  # get the name of the fill
                cv = calc_cov(data.iloc[:, i])  # calculates coefficient of variation

                item_results = {
                    "Item": data.columns[i],
                    "Client": file_name,
                    "nMAPE": round(current_nmape, 2),
                    "RMSE": round(current_rmse, 2),
                    "MAE": round(current_mae, 2),
                    "Model Type": type(model).__name__,
                    "Time Taken": round(time.time() - start_time, 3),
                    "CoV": cv,
                }

                results.append(item_results)  # append the results for each model for this item
            model_instances[model_class.__name__] = model  # store the fitted model for later use

        except Exception as e:
            # Log the error with details
            print(f"Error processing model {model_class.__name__}: {e}")

    print("Processing complete!")
    return pd.DataFrame(results), model_instances


def calc_naive_ensemble_results(
    fitted_models: list,
    baseline_model: object,
    seasonality: int,
    data: pd.DataFrame,
    series_list: list[TimeSeries],
    file_path: str,
    train_percentage: float,
    prediction_length: int,
) -> pd.DataFrame:
    """Returns a dataframe with model results for each item"""
    results = []

    for model in fitted_models:
        try:
            # Log the process start for each item and model
            model_name = model.__class__.__name__ + "_NaiveEnsemble"
            print(f"Processing model {model.__class__.__name__}...")
            start_time = time.time()  # start timer

            ensemble_models = [model, baseline_model]
            naive_ensemble_model = NaiveEnsembleModel(forecasting_models=ensemble_models, show_warnings=False)

            # Perform historical forecasting
            backtest_results = naive_ensemble_model.historical_forecasts(
                series_list,
                start=train_percentage,
                forecast_horizon=prediction_length,
                stride=1,
                retrain=True,
                verbose=True,
            )

            for i, backtest in enumerate(backtest_results):
                current_nmape = calc_nMAPE(series_list[i], backtest)  # calculates normalized MAPE
                current_rmse = rmse(series_list[i], backtest)  # calculates RMSE
                current_mae = mae(series_list[i], backtest)  # calculates MAE
                file_name = file_path.split("/")[-1][:-5]  # get the name of the fill
                cv = calc_cov(data.iloc[:, i])  # calculates coefficient of variation

                item_results = {
                    "Item": data.columns[i],
                    "Client": file_name,
                    "nMAPE": round(current_nmape, 2),
                    "RMSE": round(current_rmse, 2),
                    "MAE": round(current_mae, 2),
                    "Model Type": model_name,
                    "Time Taken": round(time.time() - start_time, 3),
                    "CoV": cv,
                }

                results.append(item_results)
        except Exception as e:
            # Log the error with details
            print(f"Error processing model {model_name}: {e}")

    print("Processing complete!")
    return pd.DataFrame(results)


def calc_regression_ensemble_results(
    fitted_models: list,
    baseline_model: object,
    seasonality: int,
    data: pd.DataFrame,
    series_list: list[TimeSeries],
    file_path: str,
    train_percentage: float,
    prediction_length: int,
) -> pd.DataFrame:
    """Returns a dataframe with model results for each item"""
    results = []

    for model in fitted_models:
        try:
            # Log the process start for each item and model
            model_name = model.__class__.__name__ + "_RegressionEnsemble"
            print(f"Processing model {model.__class__.__name__}...")
            start_time = time.time()  # start timer

            ensemble_models = [model, baseline_model]
            regression_ensemble_model = RegressionEnsembleModel(
                forecasting_models=ensemble_models, regression_train_n_points=seasonality, show_warnings=False
            )

            # Perform historical forecasting
            backtest_results = regression_ensemble_model.historical_forecasts(
                series_list,
                start=train_percentage,
                forecast_horizon=prediction_length,
                stride=1,
                retrain=True,
                verbose=True,
            )

            for i, backtest in enumerate(backtest_results):
                current_nmape = calc_nMAPE(series_list[i], backtest)  # calculates normalized MAPE
                current_rmse = rmse(series_list[i], backtest)  # calculates RMSE
                current_mae = mae(series_list[i], backtest)  # calculates MAE
                file_name = file_path.split("/")[-1][:-5]  # get the name of the fill
                cv = calc_cov(data.iloc[:, i])  # calculates coefficient of variation

                item_results = {
                    "Item": data.columns[i],
                    "Client": file_name,
                    "nMAPE": round(current_nmape, 2),
                    "RMSE": round(current_rmse, 2),
                    "MAE": round(current_mae, 2),
                    "Model Type": model_name,
                    "Time Taken": round(time.time() - start_time, 3),
                    "CoV": cv,
                }

                results.append(item_results)
        except Exception as e:
            # Log the error with details
            print(f"Error processing model {model_name}: {e}")

    print("Processing complete!")
    return pd.DataFrame(results)
