"""
Revenue forecasting for Olist.

Methods:
    1. Naive forecast
    2. Seasonal naive forecast
    3. ARIMA
    4. SARIMA

Evaluation:
    - MAE
    - RMSE
    - MAPE

Forecast horizon:
    - 6 months

Important:
    The first and last months of the Olist dataset may be incomplete.
    Boundary months are removed before forecasting to avoid training
    on partial observations.

Revenue definition:
    Delivered-order product revenue (SUM(price)).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from config import DB_PATH, REPORTS_DIR


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FORECAST_HORIZON = 6

# Number of months used as validation/test set.
TEST_MONTHS = 4

# Seasonal period for monthly data.
SEASONAL_PERIOD = 12


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection."""
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------
# 1. LOAD MONTHLY REVENUE
# ---------------------------------------------------------------------

def load_monthly_revenue(
    connection: sqlite3.Connection
) -> pd.DataFrame:
    """
    Load monthly delivered revenue from the business-performance table.
    """

    query = """
        SELECT
            month,
            revenue
        FROM monthly_business_performance
        ORDER BY month;
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    if df.empty:
        raise ValueError(
            "No monthly revenue data found."
        )

    df["month"] = pd.to_datetime(
        df["month"]
    )

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["month", "revenue"]
    )

    df = (
        df
        .set_index("month")
        .sort_index()
    )

    return df


# ---------------------------------------------------------------------
# 2. REMOVE INCOMPLETE BOUNDARY MONTHS
# ---------------------------------------------------------------------

def remove_incomplete_boundary_months(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove only genuinely incomplete boundary months.

    The raw Olist dataset starts in September 2016 and ends in
    October 2018. Since the business-performance table contains
    delivered revenue only through August 2018, August 2018 is
    treated as the final complete observed month.

    Therefore:
        - remove 2016-09
        - retain 2016-10 onward
        - retain 2018-08 as the final observed month
    """

    result = df.copy()

    if len(result) <= 2:
        raise ValueError(
            "Not enough monthly observations."
        )

    first_month = result.index.min()

    logger.info(
        "Removing incomplete first month: %s",
        first_month.strftime("%Y-%m")
    )

    # Remove only the first incomplete month.
    result = result.iloc[1:].copy()

    # Ensure continuous monthly index.
    result = result.asfreq("MS")

    return result


# ---------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------

def train_test_split_time_series(
    series: pd.Series,
    test_months: int = TEST_MONTHS
) -> tuple[pd.Series, pd.Series]:
    """
    Perform a chronological train/test split.

    No random shuffling is used because this is time-series data.
    """

    if len(series) <= test_months:
        raise ValueError(
            "Not enough observations for train/test split."
        )

    train = series.iloc[:-test_months].copy()
    test = series.iloc[-test_months:].copy()

    logger.info(
        "Training observations: %d",
        len(train)
    )

    logger.info(
        "Testing observations: %d",
        len(test)
    )

    logger.info(
        "Training period: %s to %s",
        train.index.min().strftime("%Y-%m"),
        train.index.max().strftime("%Y-%m")
    )

    logger.info(
        "Testing period: %s to %s",
        test.index.min().strftime("%Y-%m"),
        test.index.max().strftime("%Y-%m")
    )

    return train, test


# ---------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------

def calculate_mape(
    actual: pd.Series,
    predicted: pd.Series
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Zero actual values are excluded.
    """

    actual_values = np.asarray(
        actual,
        dtype=float
    )

    predicted_values = np.asarray(
        predicted,
        dtype=float
    )

    mask = actual_values != 0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    actual_values[mask]
                    - predicted_values[mask]
                )
                / actual_values[mask]
            )
        )
        * 100
    )


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series
) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE."""

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


# ---------------------------------------------------------------------
# 5. NAIVE FORECAST
# ---------------------------------------------------------------------

def naive_forecast(
    train: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Naive forecast.

    Every future value equals the last observed training value.
    """

    last_value = train.iloc[-1]

    future_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    return pd.Series(
        last_value,
        index=future_index,
        name="naive"
    )


# ---------------------------------------------------------------------
# 6. SEASONAL NAIVE
# ---------------------------------------------------------------------

def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = SEASONAL_PERIOD
) -> pd.Series:
    """
    Seasonal naive forecast.

    For monthly data with a 12-month seasonality, each forecast
    repeats the value from the same month one year earlier.

    If fewer than 12 historical observations are available,
    the method falls back to the ordinary naive forecast.
    """

    future_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    if len(train) < seasonal_period:
        logger.warning(
            "Not enough data for seasonal naive. "
            "Falling back to ordinary naive."
        )

        return pd.Series(
            train.iloc[-1],
            index=future_index,
            name="seasonal_naive"
        )

    values = []

    for i in range(horizon):

        source_position = (
            len(train)
            - seasonal_period
            + (i % seasonal_period)
        )

        values.append(
            train.iloc[source_position]
        )

    return pd.Series(
        values,
        index=future_index,
        name="seasonal_naive"
    )


# ---------------------------------------------------------------------
# 7. ARIMA
# ---------------------------------------------------------------------

def fit_arima_forecast(
    train: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Fit an ARIMA model.

    Model:
        ARIMA(1,1,1)
    """

    logger.info(
        "Fitting ARIMA(1,1,1)..."
    )

    model = ARIMA(
        train,
        order=(1, 1, 1)
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=horizon
    )

    forecast.name = "arima"

    return forecast


# ---------------------------------------------------------------------
# 8. SARIMA
# ---------------------------------------------------------------------

def fit_sarima_forecast(
    train: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Fit a seasonal ARIMA model.

    Model:
        SARIMA(1,1,1)x(1,1,1,12)
    """

    logger.info(
        "Fitting SARIMA(1,1,1)x(1,1,1,12)..."
    )

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(
        disp=False
    )

    forecast = fitted.forecast(
        steps=horizon
    )

    forecast.name = "sarima"

    return forecast


# ---------------------------------------------------------------------
# 9. EVALUATE MODELS
# ---------------------------------------------------------------------

def evaluate_models(
    train: pd.Series,
    test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate forecasting methods using the test period.
    """

    predictions: dict[str, pd.Series] = {}

    # -------------------------------------------------------------
    # Naive
    # -------------------------------------------------------------

    predictions["Naive"] = naive_forecast(
        train,
        len(test)
    )

    # Align with actual test dates.
    predictions["Naive"].index = test.index

    # -------------------------------------------------------------
    # Seasonal naive
    # -------------------------------------------------------------

    predictions["Seasonal Naive"] = (
        seasonal_naive_forecast(
            train,
            len(test)
        )
    )

    predictions["Seasonal Naive"].index = test.index

    # -------------------------------------------------------------
    # ARIMA
    # -------------------------------------------------------------

    predictions["ARIMA"] = (
        fit_arima_forecast(
            train,
            len(test)
        )
    )

    predictions["ARIMA"].index = test.index

    # -------------------------------------------------------------
    # SARIMA
    # -------------------------------------------------------------

    predictions["SARIMA"] = (
        fit_sarima_forecast(
            train,
            len(test)
        )
    )

    predictions["SARIMA"].index = test.index

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    results = []

    for model_name, prediction in predictions.items():

        metrics = calculate_metrics(
            test,
            prediction
        )

        results.append(
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"]
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAPE"
    ).reset_index(
        drop=True
    )

    return results_df


# ---------------------------------------------------------------------
# 10. FIT BEST MODEL
# ---------------------------------------------------------------------

def forecast_with_best_model(
    model_name: str,
    series: pd.Series,
    horizon: int
) -> pd.Series:
    """
    Fit the selected model on the complete cleaned series
    and forecast future months.
    """

    if model_name == "Naive":

        forecast = naive_forecast(
            series,
            horizon
        )

    elif model_name == "Seasonal Naive":

        forecast = seasonal_naive_forecast(
            series,
            horizon
        )

    elif model_name == "ARIMA":

        forecast = fit_arima_forecast(
            series,
            horizon
        )

    elif model_name == "SARIMA":

        forecast = fit_sarima_forecast(
            series,
            horizon
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    forecast = forecast.clip(
        lower=0
    )

    return forecast


# ---------------------------------------------------------------------
# 11. SAVE FORECAST
# ---------------------------------------------------------------------

def save_forecast(
    connection: sqlite3.Connection,
    forecast: pd.Series,
    model_name: str
) -> pd.DataFrame:
    """Save future forecast to SQLite and CSV."""

    forecast_df = pd.DataFrame(
        {
            "month": forecast.index,
            "forecast_revenue": forecast.values,
            "model": model_name
        }
    )

    forecast_df["month"] = (
        forecast_df["month"]
        .dt.strftime("%Y-%m")
    )

    forecast_df["forecast_revenue"] = (
        forecast_df["forecast_revenue"]
        .round(2)
    )

    forecast_df.to_sql(
        "revenue_forecast",
        connection,
        if_exists="replace",
        index=False
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast_df.to_csv(
        REPORTS_DIR / "revenue_forecast.csv",
        index=False
    )

    return forecast_df


# ---------------------------------------------------------------------
# 12. SAVE MODEL COMPARISON
# ---------------------------------------------------------------------

def save_model_comparison(
    connection: sqlite3.Connection,
    results: pd.DataFrame
) -> None:
    """Save model evaluation results."""

    results.to_sql(
        "forecast_model_comparison",
        connection,
        if_exists="replace",
        index=False
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        REPORTS_DIR / "forecast_model_comparison.csv",
        index=False
    )


# ---------------------------------------------------------------------
# 13. FORECAST PLOT
# ---------------------------------------------------------------------

def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    model_name: str
) -> None:
    """Create and save historical + forecast revenue plot."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        historical.index,
        historical.values,
        marker="o",
        label="Historical Revenue"
    )

    plt.plot(
        forecast.index,
        forecast.values,
        marker="o",
        linestyle="--",
        label=f"{model_name} Forecast"
    )

    plt.axvline(
        historical.index[-1],
        linestyle=":"
    )

    plt.title(
        "Olist Monthly Revenue Forecast"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        REPORTS_DIR
        / "revenue_forecast.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    logger.info(
        "Forecast plot saved to %s",
        output_path
    )


# ---------------------------------------------------------------------
# 14. PRINT RESULTS
# ---------------------------------------------------------------------

def print_results(
    results: pd.DataFrame,
    forecast: pd.DataFrame
) -> None:
    """Print model comparison and final forecast."""

    print("\n" + "=" * 70)
    print("FORECAST MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("6-MONTH REVENUE FORECAST")
    print("=" * 70)

    print(
        forecast.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# 15. MAIN
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete revenue forecasting pipeline."""

    logger.info("=" * 70)
    logger.info("OLIST REVENUE FORECASTING")
    logger.info("=" * 70)

    connection = get_connection()

    try:

        # -------------------------------------------------------------
        # Load
        # -------------------------------------------------------------

        logger.info(
            "Loading monthly revenue..."
        )

        df = load_monthly_revenue(
            connection
        )

        logger.info(
            "Original observations: %d",
            len(df)
        )

        # -------------------------------------------------------------
        # Remove incomplete boundaries
        # -------------------------------------------------------------

        df = remove_incomplete_boundary_months(
            df
        )

        logger.info(
            "Observations after boundary handling: %d",
            len(df)
        )

        # -------------------------------------------------------------
        # Fill missing months if any
        # -------------------------------------------------------------

        series = df["revenue"].copy()

        if series.isna().any():
            missing_months=series[
            	series.isna()
            ].index
            logger.warning(
            	"Missing calendar months detected: %s",
            	", ".join(month.strftime("%Y-%m") for month in missing_months)
            )

            logger.warning(
                "Treating missing months as zero revenue because "
                "no delivered orders were recorded for those months."
            )

            series = series.fillna(0)

        # -------------------------------------------------------------
        # Train/test split
        # -------------------------------------------------------------

        train, test = train_test_split_time_series(
            series,
            TEST_MONTHS
        )

        # -------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------

        logger.info(
            "Evaluating forecasting models..."
        )

        results = evaluate_models(
            train,
            test
        )

        # -------------------------------------------------------------
        # Select best
        # -------------------------------------------------------------

        best_model = results.iloc[0]["model"]

        logger.info(
            "Best model based on MAPE: %s",
            best_model
        )

        # -------------------------------------------------------------
        # Save model comparison
        # -------------------------------------------------------------

        save_model_comparison(
            connection,
            results
        )

        # -------------------------------------------------------------
        # Fit best model on all available cleaned data
        # -------------------------------------------------------------

        logger.info(
            "Generating %d-month forecast using %s...",
            FORECAST_HORIZON,
            best_model
        )

        forecast_series = forecast_with_best_model(
            best_model,
            series,
            FORECAST_HORIZON
        )

        # -------------------------------------------------------------
        # Save forecast
        # -------------------------------------------------------------

        forecast_df = save_forecast(
            connection,
            forecast_series,
            best_model
        )

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------

        plot_forecast(
            series,
            forecast_series,
            best_model
        )

        # -------------------------------------------------------------
        # Print
        # -------------------------------------------------------------

        print_results(
            results,
            forecast_df
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            "Revenue forecasting completed successfully."
        )

        logger.info(
            "=" * 70
        )

    finally:

        connection.close()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
