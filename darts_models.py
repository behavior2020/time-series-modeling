from typing import List

import numpy as np
from darts import TimeSeries
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)
from darts.timeseries import TimeSeries


class NaiveSeasonalGrowth(LocalForecastingModel):
    def __init__(self, K: int = 1):
        """Naive Seasonal Growth Model

        This model calculates the baseline forecast using the last year's value plus the yearly trend."""

        super().__init__()
        self.K = K

    @property
    def supports_multivariate(self) -> bool:
        return True

    def fit(self, series: TimeSeries) -> None:
        super().fit(series)
        assert series.n_samples == 1, "This model expects deterministic time series"
        assert len(series) >= 24, "The series must have at least 24 months of data."

        # Calculate trailing 12-month trend
        self.last_k_months = series[-self.K :]  # Get last k months (usually K=12)
        self.previous_2k_months = series[-self.K * 2 : -self.K]  # Get 12 months before the last 12 months

        self.trend_k_months = (
            np.mean(self.last_k_months.values()) - np.mean(self.previous_2k_months.values())
        ) / np.mean(self.previous_2k_months.values())

        # series = self.training_series
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
    ) -> np.array:
        super().predict(n, num_samples, verbose=verbose, show_warnings=show_warnings)
        forecasts = [
            self.last_k_months.values()[i % len(self.last_k_months)]
            + self.trend_k_months * self.last_k_months.values()[i % len(self.last_k_months)]
            for i in range(n)
        ]

        forecast_series = np.array(forecasts)

        return self._build_forecast_series(forecast_series)


class BasicEnsembleModel:
    def __init__(self, forecasting_models: list):
        """Basic Esemble Model
        This model averages the forecast between the best performing model and our Naive Growth Seasonal baseline model
        """

        self.forecasting_models = forecasting_models

    def fit(self, series: TimeSeries) -> None:
        self.forecasting_models = self.forecasting_models + [NaiveSeasonalGrowth(self.K)]
        for model in self.forecasting_models:
            model.fit(series)

    def predict(self, num_samples: int) -> np.array:
        forecasts = []

        for model in self.forecasting_models:
            forecast = model.predict(num_samples)
            forecast = np.array(forecast.values())
            forecasts.append(forecast)
        stacked_arrays = np.hstack(forecasts)
        mean_forecasts = np.mean(stacked_arrays, axis=1)  # get means column-wise
        return mean_forecasts
