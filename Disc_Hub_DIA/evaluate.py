import numpy as np
import matplotlib.pyplot as plt

def calculate_fdr(df, col_score):
    """
    Calculate False Discovery Rate (FDR) for both internal decoy-based and external species-based evaluations.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing identification scores, decoy labels, and protein names.
    col_score : str
        Column name for the score used to rank identifications.

    Returns:
    --------
    external_fdr_v_left : np.ndarray
        Array of external FDR thresholds used for identification count analysis.
    external_fdr_v_right : np.ndarray
        Array of external FDRs corresponding to report FDR thresholds.
    report_fdr_v : list
        List of report FDR thresholds used for evaluation.
    id_num_v : np.ndarray
        Number of unique identifications (based on 'pr_id') at each external FDR threshold.
    ids_report_fdr : int
        Number of targets identified at 1% internal (decoy-based) FDR.
    ids_external_fdr : int
        Number of targets identified at 1% external (species-based) FDR.
    """

    # FDR calculation based on decoy
    df_sorted = df.sort_values(by=col_score, ascending=False, ignore_index=True)
    target_num = (df_sorted.decoy == 0).cumsum()
    decoy_num = (df_sorted.decoy == 1).cumsum()
    target_num[target_num == 0] = 1
    df_sorted['q_pr'] = decoy_num / target_num
    df_sorted['q_pr'] = df_sorted['q_pr'][::-1].cummin()
    ids_report_fdr = sum((df_sorted.q_pr < 0.01) & (df_sorted.decoy == 0))
    print(f'Ids at report 1% FDR ({col_score}): {ids_report_fdr}')

    # Assign species based on protein names
    df_sorted = df_sorted[~df_sorted['protein_names'].isna()].copy()
    df_sorted['species'] = 'HUMAN'
    df_sorted.loc[df_sorted['protein_names'].str.contains('ARATH'), 'species'] = 'ARATH'
    df_sorted.loc[df_sorted['protein_names'].str.contains('HUMAN'), 'species'] = 'HUMAN'

    # FDR calculation based on external species (HUMAN vs ARATH)
    df_sorted = df_sorted[df_sorted['decoy'] == 0].reset_index(drop=True)
    df_sorted = df_sorted.sort_values(by=col_score, ascending=False, ignore_index=True)
    target_num = (df_sorted.species == 'HUMAN').cumsum()
    decoy_num = (df_sorted.species == 'ARATH').cumsum()
    target_num[target_num == 0] = 1
    df_sorted['q_pr_external'] = decoy_num / target_num
    df_sorted['q_pr_external'] = df_sorted['q_pr_external'][::-1].cummin()
    ids_external_fdr = sum((df_sorted.q_pr_external < 0.01) & (df_sorted.decoy == 0))
    print(f'Ids at external 1% FDR ({col_score}): {ids_external_fdr}')

    # Prepare data for FDR analysis and plotting
    fdr_v = np.arange(0.0005, 0.05, 0.001)
    external_fdr_v_left, external_fdr_v_right = [], []
    report_fdr_v = []
    id_num_v = []

    for fdr in fdr_v:
        # Internal FDR → External FDR relationship
        report_fdr_v.append(fdr)
        df_temp = df_sorted[df_sorted['q_pr'] < fdr]
        external_fdr_v_right.append(df_temp['q_pr_external'].max())

        # External FDR → ID count relationship
        external_fdr_v_left.append(fdr)
        df_temp = df_sorted[(df_sorted['q_pr_external'] < fdr) & (df_sorted.decoy == 0)]
        id_num_v.append(df_temp['pr_id'].nunique())

    # Convert results to arrays
    external_fdr_v_left = np.array(external_fdr_v_left)
    external_fdr_v_right = np.array(external_fdr_v_right)
    id_num_v = np.array(id_num_v)

    return external_fdr_v_left, external_fdr_v_right, report_fdr_v, id_num_v, ids_report_fdr, ids_external_fdr

def plot_ids_and_fdr(df, col_score, save_path):
    """
    Plot identification numbers and FDR relationships for a given score column.

    This function visualizes:
    - The relationship between external FDR thresholds and the number of identified targets (left y-axis).
    - The relationship between external FDR and report FDR (right y-axis).

    It also annotates the plot with the number of identified targets at 1% report and external FDR.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing scores, decoy labels, and species information.
    col_score : str
        Column name containing the prediction scores used for FDR calculation.
    save_path : str
        File path to save the plot image. If empty or None, the plot will not be saved to disk.
    """

    # Calculate FDR data for the given score
    external_fdr_v_left1, external_fdr_v_right1, report_fdr_v1, id_num_v1, ids_report_fdr1, ids_external_fdr1 = calculate_fdr(
        df, col_score)

    # Initialize the plot with two y-axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0, 0.05, 100), np.linspace(0, 0.05, 100), linestyle='--', color='grey')

    # Plot external vs report FDR on right y-axis (blue curve)
    ax2.plot(external_fdr_v_right1, report_fdr_v1, label=col_score, color='blue')

    # Plot external FDR vs number of IDs on left y-axis (blue thick curve)
    ax1.plot(external_fdr_v_left1, id_num_v1, label=col_score, color='blue', linewidth=3)

    # Configure axis labels and tick styles
    ax1.set_xlabel('External  FDR', fontsize=22)
    ax1.set_ylabel('#Precursors', color='black', fontsize=20)
    ax2.set_ylabel('Report  FDR', color='red', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=15)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=15)

    # Add legend
    plt.legend(fontsize=15,  loc='best')

    # Add annotation box with 1% FDR stats
    text = (
        f'Ids at report 1% FDR ({col_score}): {ids_report_fdr1}\n'
        f'Ids at external 1% FDR ({col_score}): {ids_external_fdr1}'
    )
    plt.text(0.95, 0.05, text, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")

    plt.show()
