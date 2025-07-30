from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train_lda_model(model_id, X_train, y_train, X_val):
    """
    Trains an LDA model and evaluates it on a validation set.

    Args:
        model_id (int): Identifier for the model.
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Binary training labels.
        X_val (ndarray): Validation feature matrix.

    Returns:
        model_id, val_probs, lda
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    val_probs = lda.predict_proba(X_val)[:, 1]
    return model_id, val_probs, lda

def train_lda_model_without_val(model_id, X_total, y_total):
    """
    Trains an LDA model on the entire dataset and predicts on it.

    Args:
        model_id (int): Identifier for the model.
        X_total (ndarray): Feature matrix.
        y_total (ndarray): Labels.

    Returns:
        model_id, total_probs, lda
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_total, y_total)
    total_probs = lda.predict_proba(X_total)[:, 1]
    return model_id, total_probs, lda

def train_lda_model_semi(model_id, X_train, y_train, X_val, X_unknown):
    """
    Trains an LDA model for semi-supervised learning.

    Args:
        model_id (int): Model identifier.
        X_train (ndarray): Labeled training features.
        y_train (ndarray): Labels for training.
        X_val (ndarray): Validation feature matrix.
        X_unknown (ndarray): Unlabeled feature matrix.

    Returns:
        model_id, train_probs, val_probs, unknown_probs, lda
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    train_probs = lda.predict_proba(X_train)[:, 1]
    val_probs = lda.predict_proba(X_val)[:, 1]
    unknown_probs = lda.predict_proba(X_unknown)[:, 1]
    return model_id, train_probs, val_probs, unknown_probs, lda
