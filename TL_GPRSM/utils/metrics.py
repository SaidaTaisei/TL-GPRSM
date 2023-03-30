import numpy as np

def rmse(y_true, y_pred):
    """Root mean squared error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def mae(y_true, y_pred):
    """Mean absolute error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: MAE
    """
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """Mean absolute percentage error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: MAPE
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def rmspe(y_true, y_pred):
    """Root mean squared percentage error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: RMSPE
    """
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))) * 100.0

def r2_index(y_true, y_pred):
    """R2 index
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: R2 index
    """
    return 1.0 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))

def mse(y_true, y_pred):
    """Mean squared error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: MSE
    """
    return np.mean(np.square(y_true - y_pred))

def msle(y_true, y_pred):
    """Mean squared logarithmic error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: MSLE
    """
    return np.mean(np.square(np.log(y_true + 1.0) - np.log(y_pred + 1.0)))

def rmsle(y_true, y_pred):
    """Root mean squared logarithmic error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: RMSLE
    """
    return np.sqrt(np.mean(np.square(np.log(y_true + 1.0) - np.log(y_pred + 1.0))))

def wmae(y_true, y_pred, weights):
    """Weighted mean absolute error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        weights (np.array): weights
    Returns:
        float: WMAE
    """
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def wmape(y_true, y_pred, weights):
    """Weighted mean absolute percentage error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        weights (np.array): weights
    Returns:
        float: WMAPE
    """
    return np.sum(weights * np.abs((y_true - y_pred) / y_true)) / np.sum(weights)

def mpe(y_true, y_pred):
    """Mean percentage error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: MPE
    """
    return np.mean((y_true - y_pred) / y_true) * 100.0

def me(y_true, y_pred):
    """Mean error
    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
    Returns:
        float: ME
    """
    return np.mean(y_true - y_pred)

