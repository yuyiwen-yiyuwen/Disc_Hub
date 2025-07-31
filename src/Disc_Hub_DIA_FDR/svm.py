import torch
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_svm_model(model_id, X_train, y_train, X_val):
    """
    Trains a linear SVM model with probability calibration and evaluates it on a validation set.

    This function uses an SGDClassifier with hinge loss (equivalent to a linear SVM)
    and applies Platt scaling via CalibratedClassifierCV to output probability estimates.

    Args:
        model_id (int): Identifier for the model (useful in ensemble settings).
        X_train (ndarray): Feature matrix for training data.
        y_train (ndarray): Binary labels for training data.
        X_val (ndarray): Feature matrix for validation data.

    Returns:
        model_id, val_probs, svm
        model_id: ID of the trained model.
        val_probs: Predicted probabilities for the positive class on the validation set.
        svm: Trained CalibratedClassifierCV model.
    """
    # Use an SGDClassifier with hinge loss (equivalent to linear SVM)
    base_model = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)

    # Calibrate output probabilities using Platt scaling
    svm = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

    # Fit the model on training data
    svm.fit(X_train, y_train)

    # Get predicted probabilities for the positive class on validation data
    val_probs = svm.predict_proba(X_val)[:, 1]

    return model_id, val_probs, svm

def train_svm_model_without_val(model_id, X_total, y_total):
    """
    Trains a linear SVM model with probability calibration on the entire dataset.

    This function fits an SGD-based linear SVM on all available data and outputs
    probability predictions for the same data. Useful in semi-supervised settings
    or final model training without a separate validation set.

    Args:
        model_id (int): Identifier for the model (useful for tracking in ensembles).
        X_total (ndarray): Feature matrix for the entire dataset.
        y_total (ndarray): Binary labels for the entire dataset.

    Returns:
        model_id, total_probs, svm
        model_id: ID of the trained model.
        total_probs: Predicted probabilities for the positive class on the entire dataset.
        svm: Trained CalibratedClassifierCV model.
    """
    # Use an SGDClassifier with hinge loss (equivalent to linear SVM)
    base_model = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)

    # Calibrate output probabilities using Platt scaling
    svm = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

    # Fit the model on the entire dataset
    svm.fit(X_total, y_total)

    # Get predicted probabilities for the positive class on the entire dataset
    total_probs = svm.predict_proba(X_total)[:, 1]

    return model_id, total_probs, svm

def train_svm_model_semi(model_id, X_train, y_train, X_val, X_unknown):
    """
    Trains a linear SVM model with probability calibration for semi-supervised learning.

    This function trains an SGD-based linear SVM on labeled training data, applies
    probability calibration using Platt scaling, and returns probability predictions
    for training, validation, and unknown (unlabeled) datasets.

    Args:
        model_id (int): Identifier for the model (useful for ensemble tracking).
        X_train (ndarray): Feature matrix for labeled training data.
        y_train (ndarray): Labels for the training data (binary).
        X_val (ndarray): Feature matrix for validation data.
        X_unknown (ndarray): Feature matrix for unknown or unlabeled data.

    Returns:
        model_id, train_probs, val_probs, unknown_probs, svm
        model_id: ID of the trained model.
        train_probs: Predicted probabilities for the positive class on training data.
        val_probs: Predicted probabilities on the validation data.
        unknown_probs: Predicted probabilities on the unknown (unlabeled) data.
        svm: Trained CalibratedClassifierCV model.
    """
    # Use an SGDClassifier with hinge loss (equivalent to linear SVM)
    base_model = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)

    # Calibrate output probabilities using Platt scaling
    svm = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

    # Fit the model on training data
    svm.fit(X_train, y_train)

    # Get predicted probabilities for the positive class on training, validation, and unknown data
    train_probs = svm.predict_proba(X_train)[:, 1]
    val_probs = svm.predict_proba(X_val)[:, 1]
    unknown_probs = svm.predict_proba(X_unknown)[:, 1]

    return model_id, train_probs, val_probs, unknown_probs, svm