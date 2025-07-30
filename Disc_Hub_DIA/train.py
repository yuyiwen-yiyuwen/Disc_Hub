import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import KFold
import torch.multiprocessing  as mp
import itertools
from sklearn.preprocessing import StandardScaler
from .svm import train_svm_model, train_svm_model_without_val, train_svm_model_semi
from .XGBoost import train_xgboost, train_xgboost_without_val, train_xgboost_semi
from .mlp_dia_nn import train_with_mlp_DIA_NN, train_mlp_DIA_NN_without_val, train_mlp_DIA_NN_semi
from .LDA import train_lda_model_without_val, train_lda_model_semi, train_lda_model
from .dynamic_weighting import dynamic_weighting

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a fixed random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    df[f'fully_xgboost'] = final_pred

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
    
    df[f'fully_svm'] = final_pred

    return df


def train_ensemble_model_Fully_Supervised_LDA(df, x_total, y_total, num_nn=1):
    """
    Trains an ensemble of fully supervised LDA base learners on the entire dataset and
    generates ensemble predictions with dynamic weighting.

    This function performs the following steps:
    - Sets a fixed random seed for reproducibility.
    - Standardizes the entire feature dataset.
    - Initializes an out-of-fold prediction matrix for all base learners.
    - Trains multiple LDA models in parallel without validation data.
    - Collects predictions from each base learner.
    - Computes dynamic weights based on prediction confidence (standard deviation).
    - Combines predictions using weighted averaging.
    - Normalizes final ensemble predictions between 0 and 1.
    - Adds the final ensemble probabilities as a new column to the input DataFrame.

    Args:
        df (pandas.DataFrame): Original data frame to which ensemble predictions will be added.
        x_total (ndarray): Feature matrix of all samples.
        y_total (ndarray): Labels for all samples.
        num_nn (int): Number of LDA base learners to train (default is 1).

    Returns:
        pandas.DataFrame: Input DataFrame with an additional column 'fully_lda' containing
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
                train_lda_model_without_val,
                [(j, x_scaled, y_total) for j in batch])
            results.extend(batch_results)

        # Collect predictions from each model
        for j, res in enumerate(results):
            oof_probs[:, j] = res[1].flatten()

    # Compute dynamic weights based on confidence
    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Post-processing: normalize final predictions to [0, 1]
    final_pred = (final_predictions - final_predictions.min()) / (final_predictions.max() - final_predictions.min())

    df['fully_lda'] = final_pred

    return df

def train_ensemble_model_Fully_Supervised_MLP_DIA_NN(df, x_total, y_total,
                                                     num_nn=5, batch_size=50,
                                                      seed=42):
    """
    Trains a single mlp_DIA_NN model (which is internally an ensemble) on the full dataset
    and adds the prediction probabilities to the input DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame to which ensemble predictions will be added.
        x_total (ndarray): Feature matrix for the entire dataset.
        y_total (ndarray): Binary labels for the entire dataset.
        num_nn (int): Number of MLPs inside the ensemble model.
        batch_size (int): Batch size for training.
        max_iter (int): Maximum number of iterations for training.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with added 'ensemble_prob_fully_mlp_dia_nn' column.
    """
    # Set random seed
    set_seed(seed)

    # Standardize the input features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_total)

    # Use your existing training function (which internally does ensembling)
    model_id = 0  # just one model
    _, probs = train_mlp_DIA_NN_without_val(
        model_id=model_id,
        X_total=x_scaled,
        y_total=y_total,
        num_nn=num_nn,
        batch_size=batch_size,
        seed=seed
    )

    # Normalize predicted probabilities to [0, 1]
    probs = (probs - probs.min()) / (probs.max() - probs.min())

    # Add predictions to DataFrame
    df['fully_mlp'] = probs

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

    #print(f'IDs at {fdr_threshold * 100}% FDR ({score_col}): {ids_report_fdr}')

    return df_sorted, ids_report_fdr

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
    df[f'semi_xgboost'] = best_final_pred_norm

    return df


def train_ensemble_model_Semi_Supervised_SVM(df, x_total, y_total, num_nn=1, max_iterations=50):
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
    df['semi_svm'] = best_final_pred_norm

    return df

def train_ensemble_model_Semi_Supervised_MLP_DIA_NN(
    df, x_total, y_total, num_nn=5, max_iterations=50, batch_size=50, min_delta=0.001, seed=42
):
    """
    Train a semi-supervised MLP_DIA_NN model with iterative pseudo-labeling and FDR-based confident sample selection.

    Args:
        df (pd.DataFrame): DataFrame with 'decoy' column and sample indices.
        x_total (np.ndarray): Feature matrix for all samples.
        y_total (np.ndarray): Label vector for all samples.
        num_nn (int): Number of MLP base learners in the internal ensemble.
        max_iterations (int): Maximum semi-supervised training iterations.
        batch_size (int): Batch size for training.
        patience (int): Early stopping patience.
        min_delta (float): Minimum validation loss improvement to reset patience.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Input df with added column 'semi_mlp' for predicted probabilities.
    """
    # Also import your train_mlp_DIA_NN_semi here

    set_seed(seed)

    # Keep original indices to avoid indexing issues after slicing
    df = df.copy()
    df['original_index'] = df.index

    # Initialize strat_labels: 0=decoy, 2=unknown, 1=confident positive (to update)
    df['strat_labels'] = np.where(df['decoy'] == 1, 0, 2)

    # Find best initial confident positives based on a chosen feature column (e.g. col 159)
    best_fdr_count = 0
    best_column = None
    best_confident_idx = []

    for col_idx in range(159, 160):
        temp_df = df.copy()
        temp_df['score'] = x_total[:, col_idx]
        temp_df, fdr_ids = calculate_fdr_scores(temp_df, score_col='score', decoy_col='decoy', fdr_threshold=0.01)
        if fdr_ids > best_fdr_count:
            best_fdr_count = fdr_ids
            best_column = col_idx
            best_confident_idx = temp_df.loc[(temp_df['q_pr'] < 0.01) & (temp_df['decoy'] == 0), 'original_index'].values

    df['best_initial_score'] = x_total[:, best_column]
    print(f"Best initial FDR column index: {best_column}, confident positives: {len(best_confident_idx)}")

    # Mark initial confident positives with strat_label=1
    df.loc[df['original_index'].isin(best_confident_idx), 'strat_labels'] = 1

    # Validation ratio
    val_ratio = 0.2

    # Prepare initial train/val splits based on strat_labels
    idx_0 = df.loc[df['strat_labels'] == 0, 'original_index'].to_numpy()
    idx_1 = df.loc[df['strat_labels'] == 1, 'original_index'].to_numpy()

    n_val_0 = int(len(idx_0) * val_ratio)
    n_val_1 = int(len(idx_1) * val_ratio)

    val_idx_0 = np.random.choice(idx_0, size=n_val_0, replace=False)
    val_idx_1 = np.random.choice(idx_1, size=n_val_1, replace=False)
    val_idx = np.concatenate([val_idx_0, val_idx_1])
    train_idx = np.setdiff1d(np.concatenate([idx_0, idx_1]), val_idx)

    x_train, y_train = x_total[train_idx], y_total[train_idx]
    x_val, y_val = x_total[val_idx], y_total[val_idx]

    best_fdr_ids = 0
    best_final_pred_norm = None
    original_index = df['original_index'].to_numpy()

    for iteration in range(max_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{max_iterations}")

        # Unknown samples are those not in train or val
        known_idx = np.concatenate([train_idx, val_idx])
        unknown_idx = np.setdiff1d(original_index, known_idx)
        x_unknown = x_total[unknown_idx]
        y_unknown = y_total[unknown_idx]

        # Standardize features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_unknown_scaled = scaler.transform(x_unknown)

        # Train single MLP_DIA_NN ensemble model (internal ensemble) with semi-supervised data
        model_id = 0  # single model

        # You need to implement this function similarly to train_mlp_DIA_NN_without_val but with val and early stopping
        # train_mlp_DIA_NN_semi returns (model_id, train_probs, val_probs, unknown_probs)
        _, train_probs, val_probs, unknown_probs = train_mlp_DIA_NN_semi(
            model_id=model_id,
            X_train=x_train_scaled,
            y_train=y_train,
            X_val=x_val_scaled,
            y_val=y_val,
            X_unknown=x_unknown_scaled,
            y_unknown=None,
            num_nn=num_nn,
            batch_size=batch_size,
            min_delta=min_delta,
            seed=seed,
        )

        # Collect predicted probabilities for all samples in the original order
        oof_probs = np.empty(len(x_total))
        oof_probs[:] = np.nan
        oof_probs[train_idx] = train_probs.flatten()
        oof_probs[val_idx] = val_probs.flatten()
        oof_probs[unknown_idx] = unknown_probs.flatten()

        final_pred = oof_probs
        # Save current iteration prediction
        final_pred_norm = (final_pred - np.nanmin(final_pred)) / (np.nanmax(final_pred) - np.nanmin(final_pred))

        # Create temporary DataFrame for pseudo-label updating
        tmp_df = df.copy()
        tmp_df['strat_labels_tmp'] = df['strat_labels']
        mask_not_val = ~tmp_df.index.isin(val_idx)
        tmp_df_fdr = tmp_df.loc[mask_not_val].copy()
        tmp_df_fdr['semi_prob_tmp'] = final_pred_norm[tmp_df_fdr['original_index']]
        tmp_df_fdr, _ = calculate_fdr_scores(tmp_df_fdr, score_col='semi_prob_tmp', decoy_col='decoy', fdr_threshold=0.01)

        # Find confident pseudo-positive samples to add
        confident_idx = pd.Index(
            tmp_df_fdr.loc[(tmp_df_fdr['strat_labels_tmp'] == 2) & (tmp_df_fdr['q_pr'] < 0.01), 'original_index'].values
        )

        # Update strat_labels to 1 for these new confident positives
        df.loc[df['original_index'].isin(confident_idx), 'strat_labels'] = 1

        # Evaluate overall FDR on full dataset with updated predictions
        df_full = df.copy()
        df_full['semi_prob_tmp'] = final_pred_norm
        df_full, current_fdr_ids = calculate_fdr_scores(df_full, score_col='semi_prob_tmp', decoy_col='decoy', fdr_threshold=0.01)

        print(f"Iteration {iteration + 1}: Report 1% FDR IDs = {current_fdr_ids}")

        # Early stopping if performance degrades
        if current_fdr_ids < best_fdr_ids:
            print("Early stopping triggered due to FDR degradation.")
            break
        else:
            best_fdr_ids = current_fdr_ids
            best_final_pred_norm = final_pred_norm.copy()

        # Prepare new confident samples for train/val split
        new_confident_idx = confident_idx.difference(val_idx).difference(train_idx)
        new_confident_idx = np.array(list(new_confident_idx))
        np.random.shuffle(new_confident_idx)

        n_new_val = int(len(new_confident_idx) * val_ratio)
        new_val_idx = new_confident_idx[:n_new_val]
        new_train_idx = new_confident_idx[n_new_val:]

        # Update validation and training indices
        val_idx = np.unique(np.concatenate([val_idx, new_val_idx]))
        positive_idx = df[df['strat_labels'] == 1]['original_index'].to_numpy()
        negative_idx = df[df['strat_labels'] == 0]['original_index'].to_numpy()
        train_idx = np.setdiff1d(np.concatenate([positive_idx, negative_idx]), val_idx)

        # Sort indices and select new train/val data
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        x_train, y_train = x_total[train_idx], y_total[train_idx]
        x_val, y_val = x_total[val_idx], y_total[val_idx]

    df['semi_mlp'] = best_final_pred_norm
    return df

def train_ensemble_model_Semi_Supervised_LDA(df, x_total, y_total, num_nn=1, max_iterations=50):
    set_seed(42)
    df['original_index'] = df.index
    df['strat_labels'] = np.where(df['decoy'] == 1, 0, 2)

    best_fdr_count = 0
    best_column = None
    best_confident_idx = []

    for col_idx in range(159, 160):
        df_temp = df.copy()
        df_temp['score'] = x_total[:, col_idx]
        df_temp, fdr_ids = calculate_fdr_scores(df_temp, score_col='score', decoy_col='decoy', fdr_threshold=0.01)
        if fdr_ids > best_fdr_count:
            best_fdr_count = fdr_ids
            best_column = col_idx
            best_confident_idx = df_temp.loc[(df_temp['q_pr'] < 0.01) & (df_temp['decoy'] == 0)]['original_index'].values

    df['best_initial_score'] = x_total[:, best_column]
    print(f"Best initial FDR column index: {best_column}, confident positives: {len(best_confident_idx)}")
    df.loc[df['original_index'].isin(best_confident_idx), 'strat_labels'] = 1

    all_predictions = []
    best_fdr_ids = 0
    best_final_pred_norm = None

    original_index = df['original_index'].to_numpy()
    idx_0 = df[df['strat_labels'] == 0]['original_index'].to_numpy()
    idx_1 = df[df['strat_labels'] == 1]['original_index'].to_numpy()

    val_idx = np.concatenate([
        np.random.choice(idx_0, int(len(idx_0) * 0.2), replace=False),
        np.random.choice(idx_1, int(len(idx_1) * 0.2), replace=False)
    ])
    train_idx = np.setdiff1d(np.concatenate([idx_0, idx_1]), val_idx)

    x_train, y_train = x_total[train_idx], y_total[train_idx]
    x_val, y_val = x_total[val_idx], y_total[val_idx]

    for iteration in range(max_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{max_iterations}")
        known_idx = np.concatenate([train_idx, val_idx])
        unknown_idx = np.setdiff1d(original_index, known_idx)

        x_unknown = x_total[unknown_idx]
        y_unknown = y_total[unknown_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_unknown_scaled = scaler.transform(x_unknown)

        oof_probs = np.zeros((len(x_total), num_nn))
        train_probs, val_probs, unknown_probs = [], [], []

        for model_id in range(num_nn):
            _, train_p, val_p, unknown_p, _ = train_lda_model_semi(
                model_id, x_train_scaled, y_train, x_val_scaled, x_unknown_scaled
            )
            train_probs.append(train_p)
            val_probs.append(val_p)
            unknown_probs.append(unknown_p)

        train_probs = np.mean(train_probs, axis=0)
        val_probs = np.mean(val_probs, axis=0)
        unknown_probs = np.mean(unknown_probs, axis=0)

        oof_probs[train_idx] = train_probs.reshape(-1, 1)
        oof_probs[val_idx] = val_probs.reshape(-1, 1)
        oof_probs[unknown_idx] = unknown_probs.reshape(-1, 1)

        weights = dynamic_weighting(oof_probs)
        final_pred = np.sum(oof_probs * weights, axis=1)
        all_predictions.append(final_pred.copy())

        final_pred_norm = (final_pred - final_pred.min()) / (final_pred.max() - final_pred.min())

        tmp_df = df.copy()
        tmp_df['strat_labels_tmp'] = df['strat_labels']
        tmp_df_fdr = tmp_df.loc[~tmp_df.index.isin(val_idx)].copy()
        tmp_df_fdr['semi_prob_tmp'] = final_pred_norm[tmp_df_fdr['original_index']]
        tmp_df_fdr['original_index'] = tmp_df.loc[tmp_df_fdr.index, 'original_index']
        tmp_df_fdr, _ = calculate_fdr_scores(tmp_df_fdr, score_col='semi_prob_tmp', decoy_col='decoy',
                                             fdr_threshold=0.01)
        confident_idx = pd.Index(tmp_df_fdr.loc[
            (tmp_df_fdr['strat_labels_tmp'] == 2) & (tmp_df_fdr['q_pr'] < 0.01), 'original_index'].values)
        #print(len(confident_idx))
        df.loc[df['original_index'].isin(confident_idx), 'strat_labels'] = 1

        df_full = df.copy()
        df_full['semi_prob_tmp'] = final_pred_norm
        df_full, current_fdr_ids = calculate_fdr_scores(df_full, score_col='semi_prob_tmp', decoy_col='decoy',
                                                        fdr_threshold=0.01)

        print(f"Iteration {iteration + 1}: Report 1% FDR IDs = {current_fdr_ids}")
        if current_fdr_ids <= best_fdr_ids:
            print("Early stopping triggered due to FDR performance degradation.")
            break
        else:
            best_fdr_ids = current_fdr_ids
            best_final_pred_norm = final_pred_norm.copy()

        new_confident_idx = confident_idx.difference(val_idx).difference(train_idx)
        new_confident_idx = np.array(list(new_confident_idx))
        np.random.shuffle(new_confident_idx)
        n_new_val = int(len(new_confident_idx) * 0.2)
        new_val_idx = new_confident_idx[:n_new_val]
        new_train_idx = new_confident_idx[n_new_val:]

        val_idx = np.unique(np.concatenate([val_idx, new_val_idx]))
        positive_idx = df[df['strat_labels'] == 1]['original_index'].to_numpy()
        negative_idx = df[df['strat_labels'] == 0]['original_index'].to_numpy()
        train_idx = np.setdiff1d(np.concatenate([positive_idx, negative_idx]), val_idx)

        x_train, y_train = x_total[train_idx], y_total[train_idx]
        x_val, y_val = x_total[val_idx], y_total[val_idx]

    df[f'semi_lda'] = best_final_pred_norm
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

    df['kfold_xgboost'] = final_predictions
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

    df['kfold_svm'] = final_predictions

    return df

def train_ensemble_model_kfold_mlp_dia_nn(df, x_total, y_total, num_nn=5, n_splits=5,
                                          random_state=42, batch_size=50):
    """
    "Train the mlp_DIA_NN ensemble model using K-fold cross-validation and
    return a DataFrame with the prediction column."
    """
    oof_probs = np.zeros(len(x_total))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_total)):
        print(f"\n=== Processing Fold {fold + 1}/{n_splits} ===")

        x_train, x_val = x_total[train_idx], x_total[val_idx]
        y_train, y_val = y_total[train_idx], y_total[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # 每个fold训练一个VotingClassifier集成模型
        model_id, val_probs = train_with_mlp_DIA_NN(
            fold, x_train_scaled, y_train, x_val_scaled, y_val,
            num_nn=num_nn, batch_size=batch_size, seed=random_state)

        oof_probs[val_idx] = val_probs

    df['kfold_mlp'] = oof_probs
    return df

def train_ensemble_model_kfold_LDA(df, x_total, y_total, num_nn=1, n_splits=5, random_state=42):
    """
    Train an ensemble model using k-fold cross-validation with LDA base learners.
    For each fold, a single LDA model is trained after feature scaling.
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
        Number of LDA models to train per fold; only one is used in this function (default is 1).
    n_splits : int, optional
        Number of folds for cross-validation (default is 5).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with a new column 'kfold_lda' containing the normalized ensemble predictions.
    """
    oof_probs = np.zeros((len(x_total), num_nn))  # One model per fold
    final_predictions = np.zeros(len(x_total))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_total)):
        print(f"\n=== Processing Fold {fold + 1}/{n_splits} ===")

        x_train, x_val = x_total[train_idx], x_total[val_idx]
        y_train, y_val = y_total[train_idx], y_total[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # Use LDA instead of SVM
        _, val_probs, _ = train_lda_model(1, x_train_scaled, y_train, x_val_scaled)

        oof_probs[val_idx] = val_probs.reshape(-1, 1)

    weights = dynamic_weighting(oof_probs)
    final_predictions = np.sum(oof_probs * weights, axis=1)

    # Normalize to [0, 1]
    final_predictions = (final_predictions - np.min(final_predictions)) / \
                        (np.max(final_predictions) - np.min(final_predictions))

    df['kfold_lda'] = final_predictions

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
    if framework == 'kfold' and discriminator == 'lda':
        df = train_ensemble_model_kfold_LDA(
            df,
            x_total,
            y_total,
            num_nn=1,
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
    if framework == 'kfold' and discriminator == 'mlp':
        df = train_ensemble_model_kfold_mlp_dia_nn(
            df,
            x_total,
            y_total,
            num_nn=5,
            n_splits=5,
        )

    elif framework == 'fully' and discriminator == 'lda':
        df = train_ensemble_model_Fully_Supervised_LDA(
            df,
            x_total,
            y_total,
            num_nn=1,
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
    elif framework == 'fully' and discriminator == 'mlp':
        df = train_ensemble_model_Fully_Supervised_MLP_DIA_NN(
            df,
            x_total,
            y_total,
            num_nn=5,
        )
    elif framework == 'semi' and discriminator == 'lda':
        df = train_ensemble_model_Semi_Supervised_LDA(
            df, 
            x_total, 
            y_total, 
            num_nn=1,
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
    elif framework == 'semi' and discriminator == 'mlp':
        df = train_ensemble_model_Semi_Supervised_MLP_DIA_NN(
            df,
            x_total,
            y_total,
            num_nn=5,
            max_iterations=50,
        )

    return df