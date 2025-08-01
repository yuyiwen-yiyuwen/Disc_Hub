import pandas as pd
from Disc_Hub_DIA_FDR.train import train_ensemble
from Disc_Hub_DIA_FDR.evaluate import plot_ids_and_fdr

if __name__ == '__main__':
    file_path = r"C:\Users\53458\Desktop\Dataset\Plasma_464.parquet"
    df = pd.read_parquet(file_path)

    df = train_ensemble(df=df, framework = 'kfold', discriminator = 'mlp')

    #plot
    plot_ids_and_fdr(df,
        col_score='kfold_mlp', # framework + '_' + discriminator
        save_path=r"picture_disc_hub.png")
