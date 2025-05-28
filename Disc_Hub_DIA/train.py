import copy
import torch
import torch.nn  as nn
import torch.optim  as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool
from sklearn.model_selection import KFold
from Disc_Hub.models import ResNetModel
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import torch.multiprocessing  as mp
import xgboost as xgb
import itertools

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a fixed random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Definition of the discriminator
def train_neural_network(model_id, X_train, y_train, X_val, y_val, device=device, epochs=500,
                         patience=5, min_delta=0.001, seed=42):
    """
    Trains a MLP-based binary classification model with early stopping.

    Args:
        model_id (int): Identifier for the model.
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels (binary).
        X_val (ndarray): Validation feature matrix.
        y_val (ndarray): Validation labels (binary).
        device (torch.device): Device to train the model on (CPU or GPU).
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        seed (int): Base random seed for reproducibility.

    Returns:
        model_id
        val_probs: where val_probs are predicted probabilities on validation data.
    """
    
    set_seed(seed + model_id)
    # Initialize the model, loss function, and optimizer
    model = ResNetModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert training data to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train),
                                  torch.FloatTensor(y_train).view(-1, 1))
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    # Convert validation data to PyTorch tensors
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Early stopping logic
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())  # Save the best model
        else:
            patience_counter += 1

            # Print training information
        if epoch % 3 == 0:
            print(f"Model {model_id + 1}, Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")

        # Trigger early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(best_model)  # Restore the best model
            break

    model.eval()
    with torch.no_grad():
        val_probs = model(torch.FloatTensor(X_val).to(device))
        val_probs = val_probs.flatten().cpu().numpy()

    return model_id, val_probs

def train_neural_network_without_val(model_id, X_total, y_total, device=device, epochs=1, seed=42):
    """
    Trains a MLP-based binary classification model on the full dataset without validation.

    Args:
        model_id (int): Identifier for the model (used for random seed).
        X_total (ndarray): Feature matrix for the entire dataset.
        y_total (ndarray): Labels for the entire dataset (binary).
        device (torch.device): Device to train the model on (CPU or GPU).
        epochs (int): Number of training epochs.
        seed (int): Base random seed for reproducibility.

    Returns:
        model_id, 
        probs: where probs are predicted probabilities on the entire dataset.
    """

    set_seed(seed + model_id)

    # Initialize the model, loss function, and optimizer
    model = ResNetModel(input_dim=X_total.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert total data to PyTorch tensors
    total_dataset = TensorDataset(torch.FloatTensor(X_total),
                                  torch.FloatTensor(y_total).view(-1, 1))
    total_loader = DataLoader(total_dataset,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in total_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 3 == 0:
            print(f"Model {model_id + 1}, Epoch {epoch + 1}/{epochs}, "
                  f"Loss: {running_loss / len(total_loader):.4f}")

    model.eval()
    with torch.no_grad():
        # Extract outputs and features
        probs, features = model(torch.FloatTensor(X_total).to(device), return_features=True)
        probs = probs.cpu().numpy()

    return model_id, probs

def train_neural_network_semi(model_id, X_train, y_train, X_val, y_val, x_unknown_scaled, y_unknown, device=device,
                         epochs=500,patience=5, min_delta=0.001, seed=42):
    """
    Trains a ResNet-based binary classification model using semi-supervised learning.

    This function trains on labeled data with early stopping based on a validation set,
    and then predicts probabilities for both the training data and unlabeled (unknown) data.

    Args:
        model_id (int): Identifier for the model (used to derive random seed).
        X_train (ndarray): Feature matrix for the labeled training data.
        y_train (ndarray): Labels for the training data (binary).
        X_val (ndarray): Feature matrix for the validation data.
        y_val (ndarray): Labels for the validation data (binary).
        x_unknown_scaled (ndarray): Feature matrix for unlabeled or pseudo-labeled data.
        y_unknown (ndarray): Ground truth or placeholder labels for the unknown data (not used in training).
        device (torch.device): Device to train the model on (CPU or GPU).
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement before early stopping.
        min_delta (float): Minimum change in validation loss to count as an improvement.
        seed (int): Base random seed for reproducibility.

    Returns:
        model_id, train_probs, val_probs, unknown_probs
        train_probs: Predicted probabilities on training data.
        val_probs: Predicted probabilities on validation data.
        unknown_probs: Predicted probabilities on unknown data.
    """
    set_seed(seed + model_id)

    # Initialize the model, loss function, and optimizer
    model = ResNetModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert training data to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train),
                                  torch.FloatTensor(y_train).view(-1, 1))
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)


    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    # Convert validation data to PyTorch tensors
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Early stopping logic
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())  # Save the best model
        else:
            patience_counter += 1

        # Print training information
        if epoch % 3 == 0:
            print(f"Model {model_id + 1}, Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")

        # Trigger early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(best_model)  # Restore the best model
            break

    model.eval()
    with torch.no_grad():
        train_probs = model(torch.FloatTensor(X_train).to(device), return_features=False)
        val_probs = model(torch.FloatTensor(X_val).to(device), return_features=False)
        unknown_probs = model(torch.FloatTensor(x_unknown_scaled).to(device), return_features=False)

        # Convert features to NumPy arrays
        train_probs = train_probs.cpu().numpy()
        val_probs = val_probs.cpu().numpy()
        unknown_probs = unknown_probs.cpu().numpy()

    return model_id, train_probs, val_probs, unknown_probs

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
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process
    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        early_stopping_rounds=early_stop_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=verbose_eval
    )

    # Return probability predictions only
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)

    return model_id, train_pred, val_pred, model

def train_xgboost_without_val(model_id, X_total, y_total):
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
    print(f"Training XGBoost model {model_id + 1}")

    # Convert data to DMatrix format
    dtotal = xgb.DMatrix(X_total, label=y_total)

    # Dynamic parameter configuration (with GPU acceleration if available)
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process using all data
    model = xgb.train(
        params, dtotal,
        num_boost_round=300,
        evals=[(dtotal, "total")]  # Monitor total training loss
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
    print(f"Training XGBoost model {model_id + 1}")

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dunknown = xgb.DMatrix(X_unknown, label=y_unknown)

    # Dynamic parameter configuration (with GPU acceleration if available)
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # Use histogram-based algorithm
        'device': device,
        'subsample': 1,
        'lambda': 0
    }

    # Training process
    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        early_stopping_rounds=early_stop_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=verbose_eval
    )

    # Return probability predictions only
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    unknown_pred = model.predict(dunknown)

    return model_id, train_pred, val_pred, unknown_pred, model

def dynamic_weighting(probs_matrix):
    """
    Applies dynamic weighting to an ensemble of probability predictions based on confidence.

    This function calculates weights for each model's predictions based on the standard deviation
    across models: lower standard deviation (i.e., higher confidence) results in a higher weight.
    The weights are then normalized to sum to 1 across models for each sample.

    Args:
        probs_matrix (ndarray): A 2D NumPy array of shape (n_samples, n_models) representing
                                predicted probabilities from multiple models.

    Returns:
        ndarray: A 2D array of normalized confidence-based weights of shape (n_samples, 1),
                 to be applied to the ensemble predictions.
    """
    confidence_weights = np.std(probs_matrix, axis=0, keepdims=True)
    confidence_weights = 1 / (confidence_weights + 1e-8)
    return confidence_weights / confidence_weights.sum(axis=1, keepdims=True)

def train_ensemble_model_Fully_Supervised_MLP(df, x_total, y_total, num_nn=5):
    """
    Trains an ensemble of fully supervised MLP base learners on the entire dataset and
    generates ensemble predictions with dynamic weighting.

    This function performs the following steps:
    - Sets a fixed random seed for reproducibility.
    - Standardizes the entire feature dataset.
    - Initializes an out-of-fold prediction matrix for all base learners.
    - Trains multiple neural network models in parallel without validation data.
    - Collects predictions from each base learner.
    - Computes dynamic weights based on prediction confidence (standard deviation).
    - Combines predictions using weighted averaging.
    - Normalizes final ensemble predictions between 0 and 1.
    - Adds the final ensemble probabilities as a new column to the input DataFrame.

    Args:
        df (pandas.DataFrame): Original data frame to which ensemble predictions will be added.
        x_total (ndarray): Feature matrix of all samples.
        y_total (ndarray): Labels for all samples.
        num_nn (int): Number of neural network base learners to train (default is 5).

    Returns:
        pandas.DataFrame: Input DataFrame with an additional column 'ensemble_prob' containing
                          the ensemble predicted probabilities.
    """
    set_seed(42)  # Set random seed for reproducibility

    # Standardize all data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_total)

    # Initialize out-of-fold prediction matrix
    oof_probs = np.zeros((len(x_total), num_nn))

    # Train base learners in parallel
    with Pool(processes=5) as pool:
        results = []

        # Train models in batches (here batch size equals num_nn)
        for batch in itertools.batched(range(num_nn), 5):
            batch_results = pool.starmap(
                train_neural_network_without_val,
                [(j,  x_scaled, y_total, device) for j in batch])
            results.extend(batch_results)

        # Collect predictions from each model
        for j, res in enumerate(results):
            oof_probs[:, j] = res[1].flatten()

    # Compute dynamic weights based on confidence
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: normalize final predictions to [0, 1]
    final_pred = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())
    df[f'ensemble_prob'] = final_pred

    return df

def train_ensemble_model_Fully_Supervised_XGBoost(df, x_total, y_total, num_nn=1):
    """
    Trains an ensemble of fully supervised XGBoost base learners on the entire dataset and
    generates ensemble predictions with dynamic weighting.

    This function performs the following steps:
    - Sets a fixed random seed for reproducibility.
    - Standardizes the entire feature dataset.
    - Initializes an out-of-fold prediction matrix for all base learners.
    - Trains multiple XGBoost models in parallel without validation data.
    - Collects predictions from each base learner.
    - Computes dynamic weights based on prediction confidence (standard deviation).
    - Combines predictions using weighted averaging.
    - Normalizes final ensemble predictions between 0 and 1.
    - Adds the final ensemble probabilities as a new column to the input DataFrame.

    Args:
        df (pandas.DataFrame): Original data frame to which ensemble predictions will be added.
        x_total (ndarray): Feature matrix of all samples.
        y_total (ndarray): Labels for all samples.
        num_nn (int): Number of XGBoost base learners to train (default is 1).

    Returns:
        pandas.DataFrame: Input DataFrame with an additional column 'ensemble_prob' containing
                          the ensemble predicted probabilities.
    """
    set_seed(42)  # Set random seed for reproducibility

    # Standardize all data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_total)

    # Initialize out-of-fold prediction matrix
    oof_probs = np.zeros((len(x_total), num_nn))

    # Train base learners in parallel
    with Pool(processes=3) as pool:
        results = []

        # Train models in batches
        for batch in itertools.batched(range(num_nn), 5):
            batch_results = pool.starmap(
                train_xgboost_without_val,
                [(j,  x_scaled, y_total) for j in batch])
            results.extend(batch_results)

        # Collect predictions from each model
        for j, res in enumerate(results):
            oof_probs[:, j] = res[1].flatten()

    # Compute dynamic weights based on confidence
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: normalize final predictions to [0, 1]
    final_pred = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())
    df[f'ensemble_prob'] = final_pred

    return df


def train_ensemble_model_Fully_Supervised_SVM(df, x_total, y_total, num_nn=1):
    """
    Trains an ensemble of fully supervised SVM base learners on the entire dataset and
    generates ensemble predictions with dynamic weighting.

    This function performs the following steps:
    - Sets a fixed random seed for reproducibility.
    - Standardizes the entire feature dataset.
    - Initializes an out-of-fold prediction matrix for all base learners.
    - Trains multiple SVM models in parallel without validation data.
    - Collects predictions from each base learner.
    - Computes dynamic weights based on prediction confidence (standard deviation).
    - Combines predictions using weighted averaging.
    - Normalizes final ensemble predictions between 0 and 1.
    - Adds the final ensemble probabilities as a new column to the input DataFrame.

    Args:
        df (pandas.DataFrame): Original data frame to which ensemble predictions will be added.
        x_total (ndarray): Feature matrix of all samples.
        y_total (ndarray): Labels for all samples.
        num_nn (int): Number of SVM base learners to train (default is 1).

    Returns:
        pandas.DataFrame: Input DataFrame with an additional column 'ensemble_prob' containing
                          the ensemble predicted probabilities.
    """
    set_seed(42)  # Set random seed for reproducibility

    # Standardize all data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_total)

    # Initialize out-of-fold prediction matrix
    oof_probs = np.zeros((len(x_total), num_nn))

    # Train base learners in parallel
    with Pool(processes=5) as pool:
        results = []

        # Train models in batches
        for batch in itertools.batched(range(num_nn), 5):
            batch_results = pool.starmap(
                train_svm_model_without_val,
                [(j,  x_scaled, y_total) for j in batch])
            results.extend(batch_results)

        # Collect predictions from each model
        for j, res in enumerate(results):
            oof_probs[:, j] = res[1].flatten()

    # Compute dynamic weights based on confidence
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: normalize final predictions to [0, 1]
    final_pred = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())
    
    df[f'ensemble_prob'] = final_pred

    return df

def calculate_fdr_scores(df, score_col, decoy_col='decoy', fdr_threshold=0.01):
    """
    Calculate FDR (False Discovery Rate) scores for a DataFrame based on a scoring column and a decoy indicator.

    This function performs the following steps:
    - Sorts the DataFrame by the specified score column in descending order.
    - Computes the cumulative count of target and decoy entries.
    - Calculates the provisional FDR (q-values) as the ratio of cumulative decoys to targets.
    - Applies a cumulative minimum in reverse order to ensure monotonicity of FDR estimates.
    - Counts the number of target entries passing the specified FDR threshold.
    - Prints the number of identified targets at the given FDR level.

    Args:
        df (pandas.DataFrame): Input DataFrame containing scoring and decoy columns.
        score_col (str): Name of the column containing the score values.
        decoy_col (str): Name of the column indicating decoy (1) or target (0) entries. Default is 'decoy'.
        fdr_threshold (float): Desired FDR threshold for reporting significant hits. Default is 0.01 (1%).

    Returns:
            pandas.DataFrame: Sorted DataFrame with an additional 'q_pr' column containing calculated FDR values.
            int: Number of target IDs reported below the specified FDR threshold.
    """
    # Sort the DataFrame by the score column in descending order and reset index
    df_sorted = df.sort_values(by=score_col, ascending=False, ignore_index=True)

    # Compute cumulative counts of targets and decoys
    target_num = (df_sorted[decoy_col] == 0).cumsum()
    decoy_num = (df_sorted[decoy_col] == 1).cumsum()

    # Avoid division by zero by replacing zeros with ones in target counts
    target_num = target_num.mask(target_num == 0, 1)

    # Calculate provisional FDR (q-values)
    df_sorted['q_pr'] = decoy_num / target_num

    # Apply cumulative minimum in reverse to ensure monotonicity
    df_sorted['q_pr'] = df_sorted['q_pr'][::-1].cummin()

    # Count number of target entries below the FDR threshold
    ids_report_fdr = sum((df_sorted['q_pr'] < fdr_threshold) & (df_sorted[decoy_col] == 0))

    print(f'IDs at {fdr_threshold * 100}% FDR ({score_col}): {ids_report_fdr}')

    return df_sorted, ids_report_fdr

def train_ensemble_model_Semi_Supervised_MLP(df, x_total, y_total, num_nn=5, max_iterations=50):
    """
    Train an ensemble semi-supervised MLP model with iterative pseudo-labeling and FDR-based confident sample selection.

    This function performs the following steps:
    - Initializes stratified labels based on decoy information.
    - Finds the best initial confident positive samples using FDR on one feature column.
    - Iteratively trains MLP base learners in parallel with semi-supervised training.
    - Updates pseudo-labels for unknown samples using FDR thresholding.
    - Expands the training set with newly confident positive samples after each iteration.
    - Employs early stopping based on FDR performance on the entire dataset.
    - Returns the DataFrame updated with the ensemble predicted probabilities.

    Args:
        df (pandas.DataFrame): DataFrame containing samples with 'decoy' column and original indices.
        x_total (np.ndarray): Feature matrix for all samples.
        y_total (np.ndarray): Label vector for all samples.
        num_nn (int): Number of neural networks in the ensemble. Default is 5.
        max_iterations (int): Maximum number of semi-supervised training iterations. Default is 50.

    Returns:
        pandas.DataFrame: Updated DataFrame containing an 'ensemble_prob' column with final predicted probabilities.
    """
    set_seed(42)  # Set random seed

    # Create a global index to avoid index errors after copying the DataFrame
    df['original_index'] = df.index

    # Initialize strat_labels column: 0 for decoys, 2 for unknown, 1 will be assigned to confident positives
    df['strat_labels'] = np.where(df['decoy'] == 1, 0, 2)

    # Track best initial confident samples based on FDR
    best_fdr_count = 0
    best_column = None
    best_confident_idx = []

    # Iterate over a specific feature column to find best initial confident positives (here fixed at col 159)
    for col_idx in range(159, 160):
        df_temp = df.copy()
        df_temp['score'] = x_total[:, col_idx]

        df_temp, fdr_ids = calculate_fdr_scores(df_temp, score_col='score', decoy_col='decoy', fdr_threshold=0.01)

        if fdr_ids > best_fdr_count:
            best_fdr_count = fdr_ids
            best_column = col_idx
            # Extract original indices of confident positives
            best_confident_idx = df_temp.loc[(df_temp['q_pr'] < 0.01) & (df_temp['decoy'] == 0)][
                'original_index'].values

    df['best_initial_score'] = x_total[:, best_column]
    print(f"Best initial FDR column index: {best_column}, confident positives: {len(best_confident_idx)}")

    # Label these confident samples as initial positive samples
    df.loc[df['original_index'].isin(best_confident_idx), 'strat_labels'] = 1

    # Initialize variables
    all_predictions = []
    best_fdr_ids = 0  # Best number of reported IDs at 1% FDR
    val_ratio = 0.2
    best_final_pred_norm = None

    # Explicitly save original indices
    original_index = df['original_index'].to_numpy()

    # Get original indices for negative and positive strata
    idx_0 = df.loc[df['strat_labels'] == 0, 'original_index'].to_numpy()
    idx_1 = df.loc[df['strat_labels'] == 1, 'original_index'].to_numpy()

    # Sample validation indices from negative and positive sets
    n_val_0 = int(len(idx_0) * val_ratio)
    n_val_1 = int(len(idx_1) * val_ratio)

    val_idx_0 = np.random.choice(idx_0, size=n_val_0, replace=False)
    val_idx_1 = np.random.choice(idx_1, size=n_val_1, replace=False)

    val_idx = np.concatenate([val_idx_0, val_idx_1])

    # Training indices are all except validation indices
    train_idx = np.setdiff1d(np.concatenate([idx_0, idx_1]), val_idx)

    # Select training and validation data based on original indices
    x_train, y_train = x_total[train_idx], y_total[train_idx]
    x_val, y_val = x_total[val_idx], y_total[val_idx]

    for iteration in range(max_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{max_iterations}")

        # Identify unknown samples (not in training or validation)
        known_idx = np.concatenate([train_idx, val_idx])
        unknown_idx = np.setdiff1d(original_index, known_idx)
        x_unknown = x_total[unknown_idx]
        y_unknown = y_total[unknown_idx]

        # Standardize data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_unknown_scaled = scaler.transform(x_unknown)

        # Parallel training of base learners
        oof_probs = np.zeros((len(x_total), num_nn))

        with Pool(processes=5) as pool:
            results = []

            # Train in batches of 5 models
            for batch in itertools.batched(range(num_nn), 5):
                batch_results = pool.starmap(
                    train_neural_network_semi,
                    [(j, x_train_scaled, y_train, x_val_scaled, y_val,
                      x_unknown_scaled, y_unknown, device)
                     for j in batch]
                )
                results.extend(batch_results)

            val_probs = np.hstack([res[2] for res in results])
            oof_probs[val_idx] = val_probs

            train_probs = np.hstack([res[1] for res in results])
            oof_probs[train_idx] = train_probs

            unknown_probs = np.hstack([res[3] for res in results])
            oof_probs[unknown_idx] = unknown_probs

        # Ensemble prediction fusion with dynamic weighting
        weights = dynamic_weighting(oof_probs)
        final_pred = np.sum(oof_probs * weights, axis=1)

        # Save current iteration predictions
        all_predictions.append(final_pred.copy())

        # Normalize predictions
        final_pred_norm = (final_pred - final_pred.min()) / (final_pred.max() - final_pred.min())

        # Pseudo-label updating based on 1% FDR thresholding
        tmp_df = df.copy()
        tmp_df['strat_labels_tmp'] = df['strat_labels'].copy()

        # Calculate FDR scores on non-validation samples
        mask_not_val = ~tmp_df.index.isin(val_idx)
        tmp_df_fdr = tmp_df.loc[mask_not_val].copy()
        tmp_df_fdr['semi_prob_tmp'] = final_pred_norm[tmp_df_fdr['original_index']]
        tmp_df_fdr['original_index'] = tmp_df.loc[tmp_df_fdr.index, 'original_index']
        tmp_df_fdr, _ = calculate_fdr_scores(tmp_df_fdr, score_col='semi_prob_tmp', decoy_col='decoy',
                                             fdr_threshold=0.01)

        # Select pseudo-labeled positives with strat_labels_tmp == 2 and q_pr < 0.01
        confident_idx = pd.Index(tmp_df_fdr.loc[
                                     (tmp_df_fdr['strat_labels_tmp'] == 2) & (tmp_df_fdr['q_pr'] < 0.01),
                                     'original_index'
                                 ].values)

        # Update strat_labels for these confident samples from 2 to 1 (positive)
        df.loc[df['original_index'].isin(confident_idx), 'strat_labels'] = 1

        # Evaluate overall FDR on entire dataset with updated predictions
        df_full = df.copy()
        df_full['semi_prob_tmp'] = final_pred_norm
        df_full, current_fdr_ids = calculate_fdr_scores(df_full, score_col='semi_prob_tmp', decoy_col='decoy',
                                                        fdr_threshold=0.01)

        print(f"Iteration {iteration + 1}: Report 1% FDR IDs = {current_fdr_ids}")

        # Early stopping if FDR performance degrades
        if current_fdr_ids < best_fdr_ids:
            print("Early stopping triggered due to FDR performance degradation.")
            break
        else:
            best_fdr_ids = current_fdr_ids
            best_final_pred_norm = final_pred_norm.copy()

        # Identify newly confident pseudo-labeled positives outside train and val sets
        new_confident_idx = confident_idx.difference(val_idx).difference(train_idx)

        # Shuffle and split new confident samples into validation and training sets
        new_confident_idx = np.array(list(new_confident_idx))
        np.random.shuffle(new_confident_idx)
        n_new_val = int(len(new_confident_idx) * 0.2)
        new_val_idx = new_confident_idx[:n_new_val]
        new_train_idx = new_confident_idx[n_new_val:]

        # Expand validation set with new confident validation samples
        val_idx = np.concatenate([val_idx, new_val_idx])
        val_idx = np.unique(val_idx)

        # Training set updated as all positive and negative samples minus validation set
        positive_idx = df[df['strat_labels'] == 1]['original_index'].to_numpy()
        negative_idx = df[df['strat_labels'] == 0]['original_index'].to_numpy()
        train_idx = np.concatenate([positive_idx, negative_idx])
        train_idx = np.setdiff1d(train_idx, val_idx)

        # Sort indices and select updated training and validation data
        train_idx_sorted = np.sort(train_idx)
        val_idx_sorted = np.sort(val_idx)

        x_train, y_train = x_total[train_idx_sorted], y_total[train_idx_sorted]
        x_val, y_val = x_total[val_idx_sorted], y_total[val_idx_sorted]

    # Save final ensemble prediction of the best iteration
    df[f'ensemble_prob'] = best_final_pred_norm

    return df


def train_ensemble_model_Semi_Supervised_XGBoost(df, x_total, y_total, num_nn=1, max_iterations=50):
    """
    Train an ensemble semi-supervised model using XGBoost with iterative pseudo-labeling.
    The method initializes confident positive samples by selecting the best initial FDR scoring column,
    then iteratively trains the model, updates pseudo-labels based on 1% FDR threshold, and expands the training set.
    Early stopping is applied if performance degrades.

    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame containing sample metadata, including 'decoy' labels and indices.
    x_total : numpy.ndarray
        Feature matrix for all samples (shape: [num_samples, num_features]).
    y_total : numpy.ndarray
        Label array for all samples (shape: [num_samples]).
    num_nn : int, optional
        Number of base learners in the ensemble (default is 1).
    max_iterations : int, optional
        Maximum number of iterative training cycles (default is 50).

    Returns:
    pandas.DataFrame
        The original DataFrame with added columns:
        - 'ensemble_prob': the final normalized prediction probabilities from the ensemble model.
    """

    set_seed(42)  # Set random seed

    # Create a global index to prevent indexing errors after df copy
    df['original_index'] = df.index

    # Initialize strat_labels column
    df['strat_labels'] = np.where(df['decoy'] == 1, 0, 2)

    # Record best results
    best_fdr_count = 0
    best_column = None
    best_confident_idx = []

    # Iterate over columns in x_total
    for col_idx in range(159, 160):
        df_temp = df.copy()
        df_temp['score'] = x_total[:, col_idx]

        df_temp, fdr_ids = calculate_fdr_scores(df_temp, score_col='score', decoy_col='decoy', fdr_threshold=0.01)

        if fdr_ids > best_fdr_count:
            best_fdr_count = fdr_ids
            best_column = col_idx
            # Extract original_index from df_temp
            best_confident_idx = df_temp.loc[(df_temp['q_pr'] < 0.01) & (df_temp['decoy'] == 0)][
                'original_index'].values

    df['best_initial_score'] = x_total[:, best_column]
    print(f"Best initial FDR column index: {best_column}, confident positives: {len(best_confident_idx)}")

    # Mark confident samples as initial positive samples
    df.loc[df['original_index'].isin(best_confident_idx), 'strat_labels'] = 1

    # Initialize variables
    all_predictions = []
    best_fdr_ids = 0  # Track the best number of reported 1% FDR IDs
    val_ratio = 0.2
    best_final_pred_norm = None

    # Explicitly save original indices
    original_index = df['original_index'].to_numpy()
    # Get original_index for strat_labels == 0 and 1
    idx_0 = df.loc[df['strat_labels'] == 0, 'original_index'].to_numpy()
    idx_1 = df.loc[df['strat_labels'] == 1, 'original_index'].to_numpy()

    # Sample validation set original indices
    n_val_0 = int(len(idx_0) * val_ratio)
    n_val_1 = int(len(idx_1) * val_ratio)

    val_idx_0 = np.random.choice(idx_0, size=n_val_0, replace=False)
    val_idx_1 = np.random.choice(idx_1, size=n_val_1, replace=False)

    val_idx = np.concatenate([val_idx_0, val_idx_1])

    # Get training set original indices (excluding validation indices)
    train_idx = np.setdiff1d(np.concatenate([idx_0, idx_1]), val_idx)

    # Select data from x_total / y_total using original_index
    x_train, y_train = x_total[train_idx], y_total[train_idx]
    x_val, y_val = x_total[val_idx], y_total[val_idx]

    for iteration in range(max_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{max_iterations}")

        # Handle unknown set
        known_idx = np.concatenate([train_idx, val_idx])
        unknown_idx = np.setdiff1d(original_index, known_idx)
        x_unknown = x_total[unknown_idx]
        y_unknown = y_total[unknown_idx]

        # Data standardization
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_unknown_scaled = scaler.transform(x_unknown)

        # Train base models in parallel
        oof_probs = np.zeros((len(x_total), num_nn))

        # Train base learner
        with Pool(processes=1) as pool:
            results = []

            # Batch training
            batch_results = pool.starmap(
                train_xgboost_semi,
                [(1, x_train_scaled, y_train, x_val_scaled, y_val, x_unknown_scaled, y_unknown)])
            results.extend(batch_results)

            val_probs = np.hstack([res[2] for res in results])
            oof_probs[val_idx] = val_probs.reshape(-1, 1)

            train_probs = np.hstack([res[1] for res in results])
            oof_probs[train_idx] = train_probs.reshape(-1, 1)

            unknown_probs = np.hstack([res[3] for res in results])
            oof_probs[unknown_idx] = unknown_probs.reshape(-1, 1)

        # Ensemble prediction fusion
        weights = dynamic_weighting(oof_probs)
        final_pred = np.sum(oof_probs * weights, axis=1)

        # Save current iteration predictions
        all_predictions.append(final_pred.copy())

        # Normalize predictions
        final_pred_norm = (final_pred - final_pred.min()) / (final_pred.max() - final_pred.min())

        # Pseudo-label update logic using 1% FDR threshold
        tmp_df = df.copy()
        tmp_df['strat_labels_tmp'] = df['strat_labels'].copy()

        # Compute FDR ranking on non-validation samples
        mask_not_val = ~tmp_df.index.isin(val_idx)
        tmp_df_fdr = tmp_df.loc[mask_not_val].copy()
        tmp_df_fdr['semi_prob_tmp'] = final_pred_norm[tmp_df_fdr['original_index']]
        tmp_df_fdr['original_index'] = tmp_df.loc[tmp_df_fdr.index, 'original_index']
        tmp_df_fdr, _ = calculate_fdr_scores(tmp_df_fdr, score_col='semi_prob_tmp', decoy_col='decoy',
                                             fdr_threshold=0.01)

        # Select pseudo-labeled positive samples with strat_labels_tmp == 2 and q_pr < 0.01
        confident_idx = pd.Index(tmp_df_fdr.loc[
                                     (tmp_df_fdr['strat_labels_tmp'] == 2) & (tmp_df_fdr['q_pr'] < 0.01),
                                     'original_index'
                                 ].values)

        # Update strat_labels from 2 to 1 for these samples
        df.loc[df['original_index'].isin(confident_idx), 'strat_labels'] = 1

        # Calculate 1% FDR metric on all samples (train/val/unknown)
        df_full = df.copy()
        df_full['semi_prob_tmp'] = final_pred_norm
        df_full, current_fdr_ids = calculate_fdr_scores(df_full, score_col='semi_prob_tmp', decoy_col='decoy',
                                                        fdr_threshold=0.01)

        print(f"Iteration {iteration + 1}: Report 1% FDR IDs = {current_fdr_ids}")

        # Early stopping condition
        if current_fdr_ids < best_fdr_ids:
            print("Early stopping triggered due to FDR performance degradation.")
            break
        else:
            best_fdr_ids = current_fdr_ids
            best_final_pred_norm = final_pred_norm.copy()

        new_confident_idx = confident_idx.difference(val_idx).difference(train_idx)

        # Split newly confident samples
        new_confident_idx = np.array(list(new_confident_idx))

        np.random.shuffle(new_confident_idx)
        n_new_val = int(len(new_confident_idx) * 0.2)
        new_val_idx = new_confident_idx[:n_new_val]
        new_train_idx = new_confident_idx[n_new_val:]

        # Update validation set (gradually expand)
        val_idx = np.concatenate([val_idx, new_val_idx])
        val_idx = np.unique(val_idx)

        # Training set is all samples minus validation set
        positive_idx = df[df['strat_labels'] == 1]['original_index'].to_numpy()
        negative_idx = df[df['strat_labels'] == 0]['original_index'].to_numpy()
        train_idx = np.concatenate([positive_idx, negative_idx])
        train_idx = np.setdiff1d(train_idx, val_idx)

        train_idx_sorted = np.sort(train_idx)
        val_idx_sorted = np.sort(val_idx)

        x_train, y_train = x_total[train_idx_sorted], y_total[train_idx_sorted]
        x_val, y_val = x_total[val_idx_sorted], y_total[val_idx_sorted]

    # Save final results of the last iteration
    df[f'ensemble_prob'] = best_final_pred_norm

    return df


def train_ensemble_model_Semi_Supervised_SVM(df, x_total, y_total, num_nn=5, max_iterations=50):
    """
    Train an ensemble semi-supervised model using SVM with iterative pseudo-labeling.
    The method initializes confident positive samples by selecting the best initial FDR scoring column,
    then iteratively trains the model, updates pseudo-labels based on 1% FDR threshold, and expands the training set.
    Early stopping is applied if performance degrades.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame containing sample metadata including 'decoy' labels and indices.
    x_total : numpy.ndarray
        Feature matrix for all samples (shape: [num_samples, num_features]).
    y_total : numpy.ndarray
        Label array for all samples (shape: [num_samples]).
    num_nn : int, optional
        Number of base learners in the ensemble (default is 5).
    max_iterations : int, optional
        Maximum number of iterative training cycles (default is 50).

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame with added columns:
        - 'strat_labels': updated sample labels during semi-supervised training (0=negative, 1=positive, 2=unknown).
        - 'ensemble_prob': the final normalized prediction probabilities from the ensemble model.
    """

    set_seed(42)  # Set random seed

    # Create a global index to prevent index errors after copying df
    df['original_index'] = df.index

    # Initialize strat_labels column (0 for decoy, 2 for unknown)
    df['strat_labels'] = np.where(df['decoy'] == 1, 0, 2)

    # Track best results
    best_fdr_count = 0
    best_column = None
    best_confident_idx = []

    # Iterate over each column in x_total (only column 159 here)
    for col_idx in range(159, 160):
        df_temp = df.copy()
        df_temp['score'] = x_total[:, col_idx]

        df_temp, fdr_ids = calculate_fdr_scores(df_temp, score_col='score', decoy_col='decoy', fdr_threshold=0.01)

        if fdr_ids > best_fdr_count:
            best_fdr_count = fdr_ids
            best_column = col_idx
            # Extract original_index of confident positives
            best_confident_idx = df_temp.loc[(df_temp['q_pr'] < 0.01) & (df_temp['decoy'] == 0)][
                'original_index'].values

    df['best_initial_score'] = x_total[:, best_column]
    print(f"Best initial FDR column index: {best_column}, confident positives: {len(best_confident_idx)}")

    # Mark confident samples as initial positive samples
    df.loc[df['original_index'].isin(best_confident_idx), 'strat_labels'] = 1

    # Initialize variables
    all_predictions = []
    best_fdr_ids = 0  # Record best 1% FDR IDs count
    val_ratio = 0.2
    best_final_pred_norm = None

    # Explicitly save original indices
    original_index = df['original_index'].to_numpy()
    # Get original_index for strat_labels == 0 and 1
    idx_0 = df.loc[df['strat_labels'] == 0, 'original_index'].to_numpy()
    idx_1 = df.loc[df['strat_labels'] == 1, 'original_index'].to_numpy()

    # Sample validation indices
    n_val_0 = int(len(idx_0) * val_ratio)
    n_val_1 = int(len(idx_1) * val_ratio)

    val_idx_0 = np.random.choice(idx_0, size=n_val_0, replace=False)
    val_idx_1 = np.random.choice(idx_1, size=n_val_1, replace=False)

    val_idx = np.concatenate([val_idx_0, val_idx_1])

    # Get training indices excluding validation indices
    train_idx = np.setdiff1d(np.concatenate([idx_0, idx_1]), val_idx)

    # Select data by original_index from x_total and y_total
    x_train, y_train = x_total[train_idx], y_total[train_idx]
    x_val, y_val = x_total[val_idx], y_total[val_idx]

    for iteration in range(max_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{max_iterations}")

        # Handle unknown samples
        known_idx = np.concatenate([train_idx, val_idx])
        unknown_idx = np.setdiff1d(original_index, known_idx)
        x_unknown = x_total[unknown_idx]

        # Data normalization
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_unknown_scaled = scaler.transform(x_unknown)

        # Initialize prediction storage
        oof_probs = np.zeros((len(x_total), num_nn))

        # Train base learners in parallel
        with Pool(processes=3) as pool:
            results = []

            # Batch training of ensemble members
            for batch in itertools.batched(range(num_nn), 5):
                batch_results = pool.starmap(
                    train_svm_model_semi,
                    [(j, x_train_scaled, y_train, x_val_scaled, x_unknown_scaled) for j in batch])
                results.extend(batch_results)

            val_probs = np.hstack([res[2] for res in results])
            oof_probs[val_idx] = val_probs.reshape(-1, 1)

            train_probs = np.hstack([res[1] for res in results])
            oof_probs[train_idx] = train_probs.reshape(-1, 1)

            unknown_probs = np.hstack([res[3] for res in results])
            oof_probs[unknown_idx] = unknown_probs.reshape(-1, 1)

        # Ensemble prediction weighting
        weights = dynamic_weighting(oof_probs)
        final_pred = np.sum(oof_probs * weights, axis=1)

        # Save current iteration predictions
        all_predictions.append(final_pred.copy())

        # Normalize predictions
        final_pred_norm = (final_pred - final_pred.min()) / (final_pred.max() - final_pred.min())

        # Pseudo-label update based on 1% FDR threshold
        tmp_df = df.copy()
        tmp_df['strat_labels_tmp'] = df['strat_labels'].copy()

        # Calculate FDR on non-validation set
        mask_not_val = ~tmp_df.index.isin(val_idx)
        tmp_df_fdr = tmp_df.loc[mask_not_val].copy()
        tmp_df_fdr['semi_prob_tmp'] = final_pred_norm[tmp_df_fdr['original_index']]
        tmp_df_fdr['original_index'] = tmp_df.loc[tmp_df_fdr.index, 'original_index']
        tmp_df_fdr, _ = calculate_fdr_scores(tmp_df_fdr, score_col='semi_prob_tmp', decoy_col='decoy',
                                             fdr_threshold=0.01)

        # Select pseudo-labeled positives where strat_labels_tmp == 2 and q_pr < 0.01
        confident_idx = pd.Index(tmp_df_fdr.loc[
                                     (tmp_df_fdr['strat_labels_tmp'] == 2) & (tmp_df_fdr['q_pr'] < 0.01),
                                     'original_index'
                                 ].values)

        # Update strat_labels from 2 to 1 for pseudo-positives
        df.loc[df['original_index'].isin(confident_idx), 'strat_labels'] = 1

        # Compute 1% FDR metric for all samples
        df_full = df.copy()
        df_full['semi_prob_tmp'] = final_pred_norm
        df_full, current_fdr_ids = calculate_fdr_scores(df_full, score_col='semi_prob_tmp', decoy_col='decoy',
                                                        fdr_threshold=0.01)

        print(f"Iteration {iteration + 1}: Report 1% FDR IDs = {current_fdr_ids}")

        # Early stopping if performance degrades
        if current_fdr_ids < best_fdr_ids:
            print("Early stopping triggered due to FDR performance degradation.")
            break
        else:
            best_fdr_ids = current_fdr_ids
            best_final_pred_norm = final_pred_norm.copy()

        new_confident_idx = confident_idx.difference(val_idx).difference(train_idx)

        # Split new confident samples
        new_confident_idx = np.array(list(new_confident_idx))

        np.random.shuffle(new_confident_idx)
        n_new_val = int(len(new_confident_idx) * 0.2)
        new_val_idx = new_confident_idx[:n_new_val]
        new_train_idx = new_confident_idx[n_new_val:]

        # Gradually expand validation set
        val_idx = np.concatenate([val_idx, new_val_idx])
        val_idx = np.unique(val_idx)

        # Training set is full set excluding validation set
        positive_idx = df[df['strat_labels'] == 1]['original_index'].to_numpy()
        negative_idx = df[df['strat_labels'] == 0]['original_index'].to_numpy()
        train_idx = np.concatenate([positive_idx, negative_idx])
        train_idx = np.setdiff1d(train_idx, val_idx)

        train_idx_sorted = np.sort(train_idx)
        val_idx_sorted = np.sort(val_idx)

        x_train, y_train = x_total[train_idx_sorted], y_total[train_idx_sorted]
        x_val, y_val = x_total[val_idx_sorted], y_total[val_idx_sorted]

    # Save final ensemble prediction to DataFrame
    df['ensemble_prob'] = best_final_pred_norm

    return df

def train_ensemble_model_kfold(df, x_total, y_total, num_nn=5, n_splits=5, random_state=42):
    """
    Train an ensemble model using k-fold cross-validation with multiple base learners.
    Each fold trains several base learners (e.g., neural networks) in parallel.
    Predictions from all learners across folds are combined with dynamic weighting to produce final ensemble probabilities.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame to append predictions to.
    x_total : numpy.ndarray
        Feature matrix of all samples.
    y_total : numpy.ndarray
        Label vector of all samples.
    num_nn : int, optional
        Number of base learners to train per fold (default is 5).
    n_splits : int, optional
        Number of folds for cross-validation (default is 5).
    random_state : int, optional
        Random seed for reproducible shuffling (default is 42).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with a new column 'ensemble_prob' containing the normalized ensemble predictions.
    """

    # Initialize matrix to store predictions from all base learners
    oof_probs = np.zeros((len(x_total), num_nn))
    final_predictions = np.zeros(len(x_total))

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_total)):
        print(f"\n=== Processing Fold {fold + 1}/{n_splits} ===")

        x_train, x_val = x_total[train_idx], x_total[val_idx]
        y_train, y_val = y_total[train_idx], y_total[val_idx]

        # Standardize training and validation data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # Train base learners in parallel
        with Pool(processes=num_nn) as pool:
            tasks = [(i, x_train_scaled, y_train, x_val_scaled, y_val, device)
                     for i in range(num_nn)]
            results = pool.starmap(train_neural_network, tasks)

        # Sort results by model id for consistent ordering
        results.sort(key=lambda x: x[0])
        for model_id, val_probs in results:
            oof_probs[val_idx, model_id] = val_probs

    # Compute dynamic weights and ensemble predictions
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-process: smooth normalization to [0,1]
    final_predictions = (final_predictions - np.min(final_predictions)) / \
                        (np.max(final_predictions) - np.min(final_predictions))

    df['ensemble_prob'] = final_predictions
    return df

def train_ensemble_model_kfold_XGBoost(df, x_total, y_total, num_nn=1, n_splits=5, random_state=42):
    """
    Train an ensemble model using k-fold cross-validation with XGBoost base learners.
    For each fold, multiple XGBoost models are trained in parallel.
    Predictions from all models are aggregated with dynamic weighting to form the final ensemble output.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame to which ensemble predictions will be added.
    x_total : numpy.ndarray
        Feature matrix of all samples.
    y_total : numpy.ndarray
        Label vector of all samples.
    num_nn : int, optional
        Number of XGBoost base learners to train per fold (default is 1).
    n_splits : int, optional
        Number of folds for cross-validation (default is 5).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with a new column 'ensemble_prob' containing the normalized ensemble predictions.
    """

    # Initialize prediction matrix to hold outputs from all base learners
    oof_probs = np.zeros((len(x_total), num_nn))
    final_predictions = np.zeros(len(x_total))

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_total)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Split data into training and validation sets
        x_train, x_val = x_total[train_idx], x_total[val_idx]
        y_train, y_val = y_total[train_idx], y_total[val_idx]

        # Train base learners in parallel using multiprocessing
        with Pool(processes=num_nn) as pool:
            results = pool.starmap(train_xgboost,
                                   [(i, x_train, y_train, x_val, y_val) for i in range(num_nn)])

            # Collect validation predictions
            for res in results:
                model_id, _, val_preds, _ = res
                oof_probs[val_idx, model_id] = val_preds

    # Compute dynamic weights and aggregate predictions
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: smooth normalization to [0,1]
    final_predictions = (final_predictions - np.min(final_predictions)) / \
                        (np.max(final_predictions) - np.min(final_predictions))

    df['ensemble_prob'] = final_predictions
    return df

def train_ensemble_model_kfold_SVM(df, x_total, y_total, num_nn=1, n_splits=5, random_state=42):
    """
    Train an ensemble model using k-fold cross-validation with SVM base learners.
    For each fold, a single SVM model is trained after feature scaling.
    Out-of-fold predictions are collected and aggregated with dynamic weighting
    to produce final ensemble predictions.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame to which ensemble predictions will be added.
    x_total : numpy.ndarray
        Feature matrix of all samples.
    y_total : numpy.ndarray
        Label vector of all samples.
    num_nn : int, optional
        Number of SVM models to train per fold; only one is used in this function (default is 1).
    n_splits : int, optional
        Number of folds for cross-validation (default is 5).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with a new column 'ensemble_prob' containing the normalized ensemble predictions.
    """

    # Initialize matrix to store predictions from base learners
    oof_probs = np.zeros((len(x_total), num_nn))  # Here only one model's probabilities per fold
    final_predictions = np.zeros(len(x_total))

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_total)):
        print(f"\n=== Processing Fold {fold + 1}/{n_splits} ===")

        # Split data into training and validation sets
        x_train, x_val = x_total[train_idx], x_total[val_idx]
        y_train, y_val = y_total[train_idx], y_total[val_idx]

        # Standardize the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # Train one SVM model
        _, val_probs, _ = train_svm_model(1, x_train_scaled, y_train, x_val_scaled)

        # Store validation fold predictions in out-of-fold matrix
        oof_probs[val_idx] = val_probs.reshape(-1, 1)

    # Calculate dynamic weights and aggregate predictions
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: smooth normalization to [0, 1]
    final_predictions = (final_predictions - np.min(final_predictions)) / \
                        (np.max(final_predictions) - np.min(final_predictions))

    df['ensemble_prob'] = final_predictions

    return df

def train_ensemble(df, framework='kfold', discriminator='mlp'):
    """
    Main entry function to train an ensemble model based on specified framework and discriminator type.
    Supports k-fold cross-validation, fully supervised, and semi-supervised training modes
    with different base learners: MLP, XGBoost, or SVM.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing feature score columns and a 'decoy' label column.
    framework : str, optional
        Training framework to use. Options are:
        - 'kfold' for k-fold cross-validation,
        - 'fully' for fully supervised training,
        - 'semi' for semi-supervised training. Default is 'kfold'.
    discriminator : str, optional
        Base learner type to use. Options are 'mlp', 'xgboost', or 'svm'. Default is 'mlp'.

    Returns:
    --------
    pandas.DataFrame
        Input DataFrame appended with an 'ensemble_prob' column containing ensemble prediction scores.
    """

    # Ensure proper multiprocessing start method for cross-platform compatibility
    mp.set_start_method('spawn', force=True)

    # Extract score columns (features) from DataFrame
    score_columns = [col for col in df.columns if col.startswith("score")]
    score_data = df[score_columns]
    score_matrix = score_data.to_numpy()

    # Extract decoy labels and convert to positive class labels
    decoy_data = df['decoy']
    decoy = decoy_data.to_numpy()

    # Convert to x (features) and y (labels)
    y_total = 1 - decoy
    x_total = score_matrix

    # Dispatch to appropriate training function based on framework and discriminator
    if framework == 'kfold' and discriminator == 'mlp':
        df = train_ensemble_model_kfold(
            df,
            x_total,
            y_total,
            num_nn=5,
            n_splits=5,
        )
    elif framework == 'kfold' and discriminator == 'xgboost':
        df = train_ensemble_model_kfold_XGBoost(
            df,
            x_total,
            y_total,
            num_nn=1,
            n_splits=5,
        )
    elif framework == 'kfold' and discriminator == 'svm':
        df = train_ensemble_model_kfold_SVM(
            df,
            x_total,
            y_total,
            num_nn=1,
            n_splits=5,
        )
    elif framework == 'fully' and discriminator == 'mlp':
        df = train_ensemble_model_Fully_Supervised_MLP(
            df,
            x_total,
            y_total,
            num_nn=5,
        )
    elif framework == 'fully' and discriminator == 'xgboost':
        df = train_ensemble_model_Fully_Supervised_XGBoost(
            df,
            x_total,
            y_total,
            num_nn=1,
        )
    elif framework == 'fully' and discriminator == 'svm':
        df = train_ensemble_model_Fully_Supervised_SVM(
            df, 
            x_total, 
            y_total,
            num_nn=1
        )
    elif framework == 'semi' and discriminator == 'mlp':
        df = train_ensemble_model_Semi_Supervised_MLP(
            df, 
            x_total, 
            y_total, 
            num_nn=5, 
            max_iterations=50,
        )
    elif framework == 'semi' and discriminator == 'xgboost':
        df = train_ensemble_model_Semi_Supervised_XGBoost(
            df, 
            x_total, 
            y_total, 
            num_nn=1, 
            max_iterations=50,
        )
    elif framework == 'semi' and discriminator == 'svm':
        df = train_ensemble_model_Semi_Supervised_SVM(
            df, 
            x_total, 
            y_total, 
            num_nn=1, 
            max_iterations=50,
        )

    return df
