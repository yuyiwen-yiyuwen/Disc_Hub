import torch
import numpy as np
import copy
from sklearn.metrics import log_loss
from .models import mlp_DIA_NN

# Set a fixed random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_with_mlp_DIA_NN(model_id, X_train, y_train, X_val, y_val,
                          num_nn=5, batch_size=50, seed=42):
    """
    Trains an ensemble MLPClassifier model using the mlp_DIA_NN class.

    Args:
        model_id (int): Identifier for the model (used for seeding).
        X_train (ndarray): Training features.
        y_train (ndarray): Binary training labels.
        X_val (ndarray): Validation features.
        y_val (ndarray): Binary validation labels.
        n_model (int): Number of MLPs to ensemble.
        batch_size (int): Training batch size.
        seed (int): Random seed for reproducibility.

    Returns:
        model_id: The input model identifier.
        val_probs: Predicted probabilities on validation set.
    """
    # Set global seed
    set_seed(seed + model_id)

    # Initialize and train the model
    model = mlp_DIA_NN(
        num_nn=num_nn,
        batch_size=batch_size,
        hidden_layers=(25, 20, 15, 10, 5),
        learning_rate=0.003,
        max_iter=5,
        debug=True
    )
    model.fit(X_train, y_train)

    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)[:, 1]  # Get probability of class 1

    # Optional: print validation loss
    val_loss = log_loss(y_val, val_probs)
    print(f"Model {model_id + 1}, Validation Log Loss: {val_loss:.4f}")

    return model_id, val_probs

def train_mlp_DIA_NN_without_val(model_id, X_total, y_total,
                                 num_nn=5, batch_size=50, seed=42):
    """
    Trains an ensemble MLPClassifier model (mlp_DIA_NN) on the full dataset without validation.

    Args:
        model_id (int): Identifier for the model (used for seeding).
        X_total (ndarray): Feature matrix for the entire dataset.
        y_total (ndarray): Binary labels for the entire dataset.
        num_nn (int): Number of MLPs to ensemble.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.

    Returns:
        model_id: The input model identifier.
        probs: Predicted probabilities on the entire dataset.
    """
    # Set global seed
    set_seed(seed + model_id)

    # Initialize and train the model
    model = mlp_DIA_NN(
        num_nn=num_nn,
        batch_size=batch_size,
        hidden_layers=(25, 20, 15, 10, 5),
        learning_rate=0.003,
        max_iter=1,
        debug=True
    )
    model.fit(X_total, y_total)

    # Predict probabilities on the full dataset
    probs = model.predict_proba(X_total)[:, 1]

    # Optional: print loss
    loss = log_loss(y_total, probs)
    print(f"Model {model_id + 1}, Full Data Log Loss: {loss:.4f}")

    return model_id, probs

def train_mlp_DIA_NN_semi(model_id, X_train, y_train, X_val, y_val,
                          X_unknown, y_unknown=None,
                          num_nn=5, batch_size=50,
                          min_delta=0.001, seed=42):
    """
    Semi-supervised training with ensemble MLP (mlp_DIA_NN), using early stopping based on validation loss.

    Args:
        model_id (int): Model ID for reproducibility.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels for early stopping.
        X_unknown: Unknown/unlabeled data to predict after training.
        y_unknown: Not used in training (optional placeholder).
        num_nn (int): Number of internal MLPs in the ensemble.
        batch_size (int): Training batch size.
        min_delta (float): Minimum improvement to reset patience.
        seed (int): Random seed.

    Returns:
        model_id, train_probs, val_probs, unknown_probs
    """
    set_seed(seed + model_id)

    # Initialize ensemble model
    model = mlp_DIA_NN(
        num_nn=num_nn,
        batch_size=batch_size,
        hidden_layers=(25, 20, 15, 10, 5),
        learning_rate=0.003,
        max_iter=5,
        debug=True
    )

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_loss = log_loss(y_val, val_probs)

    if val_loss < best_val_loss - min_delta:
        best_model = copy.deepcopy(model.get_model())
    else:
        patience_counter += 1

    print(f"Model {model_id + 1}, Val Loss: {val_loss:.4f}")

    # Load best model
    model.model = best_model

    # Predict
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    unknown_probs = model.predict_proba(X_unknown)[:, 1]

    return model_id, train_probs, val_probs, unknown_probs


