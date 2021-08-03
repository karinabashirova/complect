import pandas as pd
import numpy as np
import argparse
import glob as gb
import matplotlib.pyplot as plt


def create_big_df(path):
    file_names = gb.glob(path + '*.csv')

    df = pd.read_csv(file_names[0])
    columns = ['Datetime'] + list(df.columns[1:].values + str(file_names[0][-11:-9]))
    df.rename({df.columns[i]: columns[i] for i in range(len(columns))}, axis=1, inplace=True)

    # df = pd.DataFrame()
    for f in file_names[1:]:
        tmp = pd.read_csv(f)
        columns = ['Datetime'] + list(tmp.columns[1:].values + str(f[-11:-9]))
        tmp.rename({tmp.columns[i]: columns[i] for i in range(len(columns))}, axis=1, inplace=True)
        # try:
        df = df.merge(tmp, how='inner', on='Datetime')
        # except:
        #     print('ex')
        #     df = tmp

    df.dropna(inplace=True)
    df.set_index('Datetime', inplace=True)
    df_log = df.apply(lambda x: np.log(x))
    for col in df.columns:
        df_log[col] = (df_log[col] - df_log[col].mean()) / df_log[col].std()
    return df_log


def count_matrix(df_log, name):
    df_corr = df_log.corr()
    df_corr.to_csv(f'{name}corr.csv')
    # print(df_corr)

    df_z_score = pd.DataFrame()
    for i in range(len(df_log.columns)):
        df_z_score[df_log.columns[i]] = [np.nan] * len(df_log.columns)
        for j in range(len(df_log.columns)):
            x = df_log[df_log.columns[i]] - df_log[df_log.columns[j]] * df_corr[df_log.columns[i]].loc[df_log.columns[j]]
            df_z_score[df_log.columns[i]].iloc[j] = x / np.std(df_log[df_log.columns[i]]) / np.nansum(df_log[df_log.columns[j]]*df_log[df_log.columns[j]])
    print(df_z_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation matrix')
    parser.add_argument('path', help='Path to files with volatility')
    parser.add_argument('name', default="", help='Asset/prefix for name')
    args = parser.parse_args()
    print(args.path)
    df_log = create_big_df(args.path)
    count_matrix(df_log, args.name)
