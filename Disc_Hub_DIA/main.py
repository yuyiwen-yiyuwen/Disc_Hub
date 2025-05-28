import pandas as pd
from train import train_ensemble
from evaluate import plot_ids_and_fdr

if __name__ == '__main__':
    #file_path = r"E:\Py\FDR\FDR_originally\ProNN_60SPD.parquet"
    file_path = r"E:\Py\FDR\FDR_originally\plasma_600ng_464.parquet"
    df = pd.read_parquet(file_path)

    df = train_ensemble(df=df, framework = 'kfold', discriminator = 'mlp')

    #画图
    plot_ids_and_fdr(df,
        col_score1='ensemble_prob',
        col_score2='cscore_pr_run',
        save_path=r"picture_disc_hub.png")
