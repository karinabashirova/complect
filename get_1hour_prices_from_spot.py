import os
import pandas as pd
import numpy as np
import datetime
import glob as gb
import argparse
from lib.option_formulas import price_by_BS, vega, gamma, delta
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


def get_data():
    df = pd.read_csv('BTC_last_12_months.csv', usecols=['exchange time', 'open'])
    df['exchange time'] = [datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S')
                           for d in df['exchange time']]
    first_date = datetime.datetime.strptime('2020-12-26 06:59:32', '%Y-%m-%d %H:%M:%S')
    last_date = datetime.datetime.strptime('2021-06-16 08:57:45', '%Y-%m-%d %H:%M:%S')
    df = df[(df['exchange time'] <= last_date) & (df['exchange time'] >= first_date)]
    df = df[df['exchange time'].dt.minute == 0]
    df.rename(columns={'exchange time': 'date', 'open': 'spot'}, inplace=True)
    df.reset_index(inplace=True)
    df = df[['date', 'spot']]
    return df


def add_volatility(df):
    df['volatility'] = np.abs(np.log(df.spot / df.spot.shift(1))) * np.sqrt(365 * 24)
    plt.plot(df.volatility)
    plt.show()
    df.dropna(inplace=True)
    return df


def add_price_gamma(df):
    time_before_expiration = 1 / 24 / 365  # VP4 тут я решила, что дельта особо не меняется, а минутные цены не нужны)
    # time_before_expiration = 1 / 365  # VP5
    df['price'] = np.nan
    df['delta'] = np.nan
    df['gamma'] = np.nan
    for i in range(len(df)):
        print(df.date.iloc[i])
        if df.date.iloc[i].hour == 8:  # VP5
            expiration = df.date.iloc[i] + datetime.timedelta(days=1)
        else:  # VP5
            time_before_expiration = (expiration - df.date.iloc[i]).total_seconds() / (60 * 60 * 24 * 365)
        df.price.iloc[i] = price_by_BS(df.spot.iloc[i], df.spot.iloc[i],
                                       time_before_expiration,
                                       df.volatility.iloc[i], 'CALL')
        df['delta'].iloc[i] = delta(df.spot.iloc[i], df.spot.iloc[i],
                                    time_before_expiration,
                                    df.volatility.iloc[i], 'CALL')
        df['gamma'].iloc[i] = gamma(df.spot.iloc[i], df.spot.iloc[i],
                                    time_before_expiration,
                                    df.volatility.iloc[i])
    return df


def add_payout(df):
    df['payout'] = df.spot - df.spot.shift(1)
    return df


if __name__ == '__main__':
    df_ = get_data()
    df_ = add_volatility(df_)
    df_ = add_price_gamma(df_)
    df_ = add_payout(df_)
    print(df_)
    # df_.to_csv('vp5_part.csv', index=False)
    df_.to_csv('vp4_part.csv', index=False)
