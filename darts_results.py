# Import libraries
import os
import warnings

import pandas as pd
from darts import TimeSeries
from darts_config import *
from darts_functions import (
    calc_naive_ensemble_results,
    calc_nn_results,
    calc_regression_ensemble_results,
    calc_statistical_results,
    count_results,
    count_zeroes,
    create_csv,
    get_relevent_columns,
    initialize_df,
    initialize_models,
    preprocess_data,
)
from darts_models import NaiveSeasonalGrowth
from tabulate import tabulate

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=FutureWarning)


if __name__ == "__main__":
    # Load and preprocess data frame
    file_path, df = initialize_df()

    # Initalize time series objects for darts
    all_series = [TimeSeries.from_dataframe(df, value_cols=col) for col in df.columns]
    print(f"Dataset includes {len(all_series)} products")
    TRAIN_WINDOW = len(df) - VAL_WINDOW  # set train window for backtesting

    # Initialize models
    stat_models, deep_learning_models = initialize_models()

    # Calculate results for models
    stat_results, stat_instances = calc_statistical_results(
        model_list=stat_models,
        data=df,
        series_list=all_series,
        file_path=file_path,
        train_percentage=TRAIN_WINDOW,
        prediction_length=FORECAST_PERIODS,
    )

    nn_results, nn_instances = calc_nn_results(
        model_list=deep_learning_models,
        data=df,
        series_list=all_series,
        file_path=file_path,
        train_percentage=TRAIN_PERCENTAGE,
    )

    naive_ensemble_results = calc_naive_ensemble_results(
        fitted_models=list(stat_instances.values()),
        baseline_model=NaiveSeasonalGrowth(K=SEASONALITY),
        seasonality=SEASONALITY,
        data=df,
        series_list=all_series,
        file_path=file_path,
        train_percentage=TRAIN_WINDOW,
        prediction_length=FORECAST_PERIODS,
    )

    regression_ensemble_results = calc_regression_ensemble_results(
        fitted_models=list(stat_instances.values()),
        baseline_model=NaiveSeasonalGrowth(K=12),
        seasonality=SEASONALITY,
        data=df,
        series_list=all_series,
        file_path=file_path,
        train_percentage=TRAIN_WINDOW,
        prediction_length=FORECAST_PERIODS,
    )

    # Combine results
    merged_df = pd.concat(
        [stat_results, nn_results, naive_ensemble_results, regression_ensemble_results], ignore_index=True
    )
    print(tabulate(merged_df.head(), headers="keys", tablefmt="psql"))

    # Create csv of model results
    create_csv(merged_df, file_path, "summary")
    print(f"CSV file saved succfully at: {os.getcwd()}")

    # Counts number of times a model won
    model_counts = count_results(merged_df)
    print(model_counts)
