from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class RegressionMetrics:
    """
    Store regression evaluation metrics.

    Attributes
    ----------
    mae : float
        Mean absolute error.

    rmse : float
        Root mean squared error.

    r2 : float
        Coefficient of determination.
    """

    mae: float
    rmse: float
    r2: float


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_predicted: pd.Series | np.ndarray,
) -> RegressionMetrics:
    """
    Calculate evaluation metrics for regression predictions.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        Measured target values.

    y_predicted : pd.Series | np.ndarray
        Target values predicted by a regression model.

    Returns
    -------
    RegressionMetrics
        Object containing MAE, RMSE, and R-squared.

    Raises
    ------
    ValueError
        If the inputs are empty, have different lengths, or contain
        non-finite values.

    Examples
    --------
    >>> metrics = calculate_regression_metrics(
    ...     y_test,
    ...     baseline_predictions,
    ... )
    >>> print(metrics.mae)
    0.02
    >>> print(metrics.rmse)
    0.03
    """

    # Convert pandas objects to one-dimensional NumPy arrays.
    measured = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_predicted, dtype=float).reshape(-1)

    if measured.size == 0 or predicted.size == 0:
        raise ValueError("Measured and predicted values must not be empty.")

    if measured.size != predicted.size:
        raise ValueError(
            "Measured and predicted values must have the same length."
        )

    if not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        raise ValueError(
            "Measured and predicted values must contain only finite values."
        )

    # MAE gives the average absolute difference between prediction and reality.
    mae = mean_absolute_error(measured, predicted)

    # RMSE gives more weight to larger prediction errors.
    mse = mean_squared_error(measured, predicted)
    rmse = np.sqrt(mse)

    # R-squared measures performance relative to predicting the test-data mean.
    r2 = r2_score(measured, predicted)

    return RegressionMetrics(
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
    )

