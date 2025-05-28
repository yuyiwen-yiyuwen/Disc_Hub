import pandas as pd
from Disc_Hub_DIA import train_ensemble, plot_ids_and_fdr

if __name__ == '__main__':
    file_path = r"Dataset\SC_3116.parquet"
    df = pd.read_parquet(file_path)

    df = train_ensemble(df=df, framework = 'kfold', discriminator = 'xgboost')

    plot_ids_and_fdr(df,
    col_score='ensemble_prob',
    save_path=r"picture_disc_hub.png")
