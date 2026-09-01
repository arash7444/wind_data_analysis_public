import numpy as np
import pandas as pd



def prepare_ti_regression_data(
    ti_raw: pd.DataFrame,
    target_height: float,
) -> tuple[pd.DataFrame, float]:

    """
    Prepare the data for TI regression by selecting the target height and removing invalid data.
    
    Parameters
    -----------
    ti_raw : pd.DataFrame
        DataFrame containing the raw TI data.
    target_height : float
        Target height for the TI data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the prepared TI data.
    float
        Selected height.

    Examples
    --------
    >>> from wind_data_analysis.ML import prepare_ti_regression_data
    >>> data, height = prepare_ti_regression_data(
    ...     ti_raw,
    ...     target_height=120.0,
    ... )
    >>> print(height)
    120.0
    >>> print(data.head())
    
    


    """ 


    required_columns = ["Time", "height", "wind_speed", "ti"]

    missing_columns = set(required_columns) - set(ti_raw.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = ti_raw[required_columns].copy()

    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")

    available_heights = data["height"].dropna().unique()

    if len(available_heights) == 0:
        raise ValueError("No valid measurement heights were found.")

    selected_height = min(
        available_heights,
        key=lambda height: abs(height - target_height),
    )

    data = data[data["height"] == selected_height].copy()

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        subset=["Time", "wind_speed", "ti"]
    )

    data = data[
        (data["wind_speed"] > 0)
        & (data["ti"] >= 0)
        & (data["ti"] < 1)
    ]

    data = data.sort_values("Time").reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable TI observations remain after cleaning.")

    return data, float(selected_height)


def chronological_split(
    data: pd.DataFrame,
    test_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if data.empty:
        raise ValueError("Cannot split an empty dataset.")

    if not data["Time"].is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically.")

    split_index = int(len(data) * (1 - test_fraction))

    if split_index == 0 or split_index == len(data):
        raise ValueError("Not enough observations for the requested split.")

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return train_data, test_data



def create_ml_inputs(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create the feature and target datasets used by scikit-learn.

    ML V1 uses mean wind speed as the only input feature and turbulence
    intensity as the regression target.

    Parameters
    ----------
    train_data : pd.DataFrame
        Earlier observations used to train the model. The DataFrame
        must contain ``wind_speed`` and ``ti`` columns.

    test_data : pd.DataFrame
        Later observations used to evaluate the model. The DataFrame
        must contain ``wind_speed`` and ``ti`` columns.

    Returns
    -------
    X_train : pd.DataFrame
        Training features. Contains one column: ``wind_speed``.

    X_test : pd.DataFrame
        Test features. Contains one column: ``wind_speed``.

    y_train : pd.Series
        Training target containing measured TI values.

    y_test : pd.Series
        Test target containing measured TI values.

    Raises
    ------
    ValueError
        If either input DataFrame is empty or does not contain the
        required columns.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = create_ml_inputs(
    ...     train_data,
    ...     test_data,
    ... )
    >>> print(X_train.shape)
    (201, 1)
    >>> print(y_train.shape)
    (201,)
    """

    required_columns = {"wind_speed", "ti"}

    if train_data.empty or test_data.empty:
        raise ValueError("Training and test datasets must not be empty.")

    # Check both datasets before selecting the feature and target.
    for dataset_name, dataset in [
        ("train_data", train_data),
        ("test_data", test_data),
    ]:
        missing_columns = required_columns - set(dataset.columns)

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    # Double brackets create a two-dimensional feature DataFrame.
    X_train = train_data[["wind_speed"]].copy()
    X_test = test_data[["wind_speed"]].copy()

    # Single brackets create a one-dimensional target Series.
    y_train = train_data["ti"].copy()
    y_test = test_data["ti"].copy()

    return X_train, X_test, y_train, y_test

