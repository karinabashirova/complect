import os
import pandas as pd
import numpy as np
import datetime
import glob as gb
import argparse
from lib.option_formulas import price_by_BS, vega, gamma, delta

import warnings

warnings.filterwarnings("ignore")


def get_data():
    df = pd.read_csv('BTC_last_12_months.csv', usecols=['exchange time', 'open'])
    df['exchange time'] = [datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S')
                           for d in df['exchange time']]
    first_date = datetime.datetime.strptime('2020-12-26 06:59:32', '%Y-%m-%d %H:%M:%S')
    last_date = datetime.datetime.strptime('2021-06-16 08:57:45', '%Y-%m-%d %H:%M:%S')
    df = df[(df['exchange time'] <= last_date) & (df['exchange time'] >= first_date)]
    df.rename(columns={'exchange time': 'date', 'open': 'spot'}, inplace=True)
    df.reset_index(inplace=True)
    df = df[['date', 'spot']]

    df['delta_spot'] = df.spot - df.spot.shift(1)
    # print(df)
    return df


def convert_results(df):
    df.loc[:, 'delta'] = pd.to_numeric(df['delta']).copy()
    # print(df)
    df.loc[:, 'date'] = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S')
                         for d in df.date]

    # print(df_)
    return df


def get_options_data():
    # opt_df = pd.read_csv('deribit_part_vp5.csv')
    opt_df = pd.read_csv('deribit_part_vp4.csv')
    opt_df = convert_results(opt_df)
    return opt_df


def join(df, opt_df):
    opt_df['hedging_part'] = np.nan
    for i in range(len(opt_df)):
        todays_date = opt_df.date.iloc[i]
        prev_date = todays_date - datetime.timedelta(days=1)
        prev_date = prev_date.replace(hour=8, minute=0, second=0)
        short_df = df[(df.date < todays_date) & (df.date >= prev_date)]
        sum_delta_spot = short_df.delta_spot.sum()
        opt_df.hedging_part.iloc[i] = opt_df.options_count.iloc[i] * opt_df.delta.iloc[i] * sum_delta_spot

    return opt_df


if __name__ == '__main__':
    df_ = get_data()
    opt_df_ = get_options_data()
    opt_df_ = convert_results(opt_df_)
    # print(opt_df_)
    result_df = join(df_, opt_df_)
    result_df.to_csv('options_result_df_vp4.csv', index=False)
    # result_df.to_csv('options_result_df_vp5.csv', index=False)
