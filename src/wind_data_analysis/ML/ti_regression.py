import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[DummyRegressor, np.ndarray]:
    """
    Train a mean-value baseline and predict TI for the test period.

    The baseline does not learn a relationship between wind speed and TI.
    Instead, it calculates the mean TI from the training data and uses that
    value for every test prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features containing the ``wind_speed`` column.

    y_train : pd.Series
        Measured TI values for the training period.

    X_test : pd.DataFrame
        Test features for which TI predictions will be generated.

    Returns
    -------
    baseline_model : DummyRegressor
        Fitted mean-value baseline model.

    baseline_predictions : np.ndarray
        Predicted TI values for the test period. Every value will equal
        the mean TI of the training period.

    Raises
    ------
    ValueError
        If any input is empty or if the number of training features does
        not match the number of training target values.

    Examples
    --------
    >>> baseline_model, baseline_predictions = train_baseline_model(
    ...     X_train,
    ...     y_train,
    ...     X_test,
    ... )
    >>> baseline_predictions.shape
    (87,)
    >>> np.unique(baseline_predictions).size
    1
    """

    if X_train.empty or y_train.empty or X_test.empty:
        raise ValueError(
            "Training features, training target, and test features "
            "must not be empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of samples."
        )

    # Create a baseline that uses the mean training TI as its prediction.
    baseline_model = DummyRegressor(strategy="mean")

    # Fit calculates and stores the mean value of y_train.
    baseline_model.fit(X_train, y_train)

    # Generate one baseline prediction for every test observation.
    baseline_predictions = baseline_model.predict(X_test)

    return baseline_model, baseline_predictions






def train_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_neighbors: int = 5,
) -> tuple[Pipeline, np.ndarray]:
    """
    Train a KNN regression model and predict TI for the test period.

    Before training KNN, wind speed is standardized using statistics
    calculated only from the training data. For each test observation,
    KNN finds the nearest training wind speeds and returns the average
    TI of those neighbours.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features containing the ``wind_speed`` column.

    y_train : pd.Series
        Measured TI values for the training period.

    X_test : pd.DataFrame
        Test features for which TI predictions will be generated.

    n_neighbors : int, optional
        Number of nearest training observations used for each prediction.
        Default is 5.

    Returns
    -------
    knn_model : Pipeline
        Fitted scikit-learn pipeline containing the scaler and the
        KNN regression model.

    knn_predictions : np.ndarray
        TI values predicted for the test period.

    Raises
    ------
    ValueError
        If an input is empty, the training feature and target lengths
        differ, or the number of neighbours is invalid.

    Examples
    --------
    >>> knn_model, knn_predictions = train_knn_model(
    ...     X_train,
    ...     y_train,
    ...     X_test,
    ...     n_neighbors=5,
    ... )
    >>> knn_predictions.shape
    (87,)
    """

    if X_train.empty or y_train.empty or X_test.empty:
        raise ValueError(
            "Training features, training target, and test features "
            "must not be empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of samples."
        )

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    if n_neighbors > len(X_train):
        raise ValueError(
            "n_neighbors cannot be larger than the training dataset."
        )

    # The pipeline ensures that scaling is fitted only on training data.
    knn_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                KNeighborsRegressor(n_neighbors=n_neighbors),
            ),
        ]
    )

    # Fit first scales the training wind speeds and then trains KNN.
    knn_model.fit(X_train, y_train)

    # The same training-data scaling is applied before test predictions.
    knn_predictions = knn_model.predict(X_test)

    return knn_model, knn_predictions