import torch
import xgboost as xgb

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_xgboost(model_id, X_train, y_train, X_val, y_val, early_stop_rounds=10, verbose_eval=50):
    """
    Trains an XGBoost base learner for binary classification and returns probability predictions.

    This function configures and trains an XGBoost model using specified training and
    validation data. GPU acceleration is enabled if available. Only probability outputs
    are returned (no class labels).

    Args:
        model_id (int): Identifier for the model (useful for tracking in ensembles).
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
        X_val (ndarray): Validation feature matrix.
        y_val (ndarray): Validation labels.
        early_stop_rounds (int): Number of rounds to wait before early stopping if no improvement.
        verbose_eval (int): Frequency of printed evaluation metrics during training.

    Returns:
        model_id, train_pred, val_pred, model
        model_id: ID of the trained model.
        train_pred: Probability predictions on training data.
        val_pred: Probability predictions on validation data.
        model: Trained XGBoost model object.
    """
    print(f"Training XGBoost model {model_id + 1}")

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Dynamic parameter configuration (with GPU acceleration if available)
    params = {
        'max_depth': 3,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        early_stopping_rounds=early_stop_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=verbose_eval
    )

    # Return probability predictions only
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)

    return model_id, train_pred, val_pred, model

def train_xgboost_without_val(model_id, X_total, y_total, verbose_eval=100):
    """
    Trains an XGBoost base learner using all available labeled data without a validation set.

    This function trains an XGBoost model for binary classification using the full dataset
    (no validation split), and returns probability predictions for the input data.
    GPU acceleration is enabled if available.

    Args:
        model_id (int): Identifier for the model (useful for ensemble tracking).
        X_total (ndarray): Feature matrix for all labeled data.
        y_total (ndarray): Labels for all data points (binary).

    Returns:
        model_id, total_pred, model
        model_id: ID of the trained model.
        total_pred: Probability predictions for the full dataset.
        model: Trained XGBoost model object.
    """

    # Convert data to DMatrix format
    dtotal = xgb.DMatrix(X_total, label=y_total)

    # Dynamic parameter configuration (with GPU acceleration if available)
    params = {
        'max_depth': 3,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process using all data
    model = xgb.train(
        params, dtotal,
        num_boost_round=500,
        evals=[(dtotal, "total")],  # Monitor total training loss
        verbose_eval = verbose_eval
    )

    # Return probability predictions only
    total_pred = model.predict(dtotal)

    return model_id, total_pred, model

def train_xgboost_semi(model_id, X_train, y_train, X_val, y_val, X_unknown, y_unknown, early_stop_rounds=10, verbose_eval=50):
    """
    Trains an XGBoost base learner for semi-supervised learning and returns probability predictions.

    This function trains an XGBoost model for binary classification using labeled training data,
    performs validation for early stopping, and also predicts probabilities on an unlabeled (or pseudo-labeled)
    unknown set. GPU acceleration is enabled if available.

    Args:
        model_id (int): Identifier for the model (useful for ensemble tracking).
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
        X_val (ndarray): Validation feature matrix.
        y_val (ndarray): Validation labels.
        X_unknown (ndarray): Unlabeled or pseudo-labeled data feature matrix.
        y_unknown (ndarray): Pseudo-labels for unknown data (used for prediction monitoring).
        early_stop_rounds (int): Number of rounds to wait before triggering early stopping.
        verbose_eval (int): Frequency of printed evaluation logs during training.

    Returns:
        model_id, train_pred, val_pred, unknown_pred, model
        model_id: ID of the trained model.
        train_pred: Probability predictions on the training data.
        val_pred: Probability predictions on the validation data.
        unknown_pred: Probability predictions on the unknown (pseudo-labeled) data.
        model: Trained XGBoost model object.
    """

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dunknown = xgb.DMatrix(X_unknown, label=y_unknown)

    # Dynamic parameter configuration (with GPU acceleration if available)
    params = {
        'max_depth': 3,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        early_stopping_rounds=early_stop_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=verbose_eval
    )

    # Return probability predictions only
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    unknown_pred = model.predict(dunknown)

    return model_id, train_pred, val_pred, unknown_pred, model