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


def get_data_from_files(result_df, path, col_case):
    if col_case == 'new':
        cols_to_use = ['k', 'ask_c', 'exp', 'q', 'u_price']
        spot_col = 'u_price'
        exp_col = 'exp'
    else:
        cols_to_use = ['k', 'ask_c', 'e', 'q', 's0']
        spot_col = 's0'
        exp_col = 'e'
    file_names = gb.glob(path + '*.csv')
    for file in file_names:
        df = pd.read_csv(file, usecols=cols_to_use)

        def cut_df(result_df, df, time, prev_time):
            prev_spot = df[[df.q.iloc[i][11:13] == prev_time for i in range(len(df))]][spot_col].iloc[0]
            df = df[[df.q.iloc[i][11:13] == time for i in range(len(df))]]
            df.reset_index(inplace=True)

            df = convert_results(df, spot_col, exp_col, col_case)

            idx = abs(df.k - df[spot_col].iloc[0]).idxmin()
            df = df.iloc[[idx]]
            HV = np.abs(np.log(df[spot_col].iloc[0] / float(prev_spot))) * np.sqrt(365 * 24)
            result_df = result_df.append({'date': df.q.iloc[0],
                                          'expiration': df[exp_col].iloc[0],
                                          'time_before_expiration': df['time_before_expiration'].iloc[0],
                                          'weekday': df['weekday'].iloc[0],
                                          'expiration_weekday': df['expiration_weekday'].iloc[0],
                                          'spot': df[spot_col].iloc[0],
                                          'strike': df.k.iloc[0],
                                          'price': df.ask_c.iloc[0],
                                          'volatility': HV}, ignore_index=True)
            return result_df

        try:
            result_df = cut_df(result_df, df, '07', '06')
        except IndexError:
            try:
                result_df = cut_df(result_df, df, '06', '05')
            except IndexError:
                print('Empty', file)
                pass
    return result_df


weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}


def convert_results(df_, spot_col, exp_col, col_case):
    df_.loc[:, spot_col] = pd.to_numeric(df_[spot_col]).copy()
    df_.loc[:, 'k'] = pd.to_numeric(df_['k']).copy()
    df_.loc[:, 'ask_c'] = pd.to_numeric(df_['ask_c']).copy()
    df_.loc[:, 'q'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                       for d in df_.q]
    df_.loc[:, exp_col] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                           for d in df_[exp_col]]
    if col_case == 'new':
        df_['ask_c'] = df_.ask_c * df_[spot_col]

    diff = df_[exp_col] - df_.q
    times = []
    df_['weekday'] = np.nan
    df_['expiration_weekday'] = np.nan
    for i in range(len(diff)):
        df_.weekday.iloc[i] = weekdays[df_.q.iloc[i].weekday()]
        df_.expiration_weekday.iloc[i] = weekdays[df_[exp_col].iloc[i].weekday()]
        times.append(diff.iloc[i].total_seconds() / (365 * 24 * 60 * 60))
    df_['time_before_expiration'] = times
    df_ = df_[
        (df_.time_before_expiration < 7 * 24 * 60 * 60 / (365 * 24 * 60 * 60) + 3 * 60 * 60 / (365 * 24 * 60 * 60))
        & (df_.expiration_weekday == 'Friday')]
    df_.reset_index(inplace=True)
    # print(df_)
    return df_


def add_delta_gamma(df_):
    df_['gamma'] = [gamma(df_.spot.iloc[i], df_.spot.iloc[i],
                          df_.time_before_expiration.iloc[i],
                          df_.volatility.iloc[i])
                    for i in range(len(df_))]
    plt.plot(df_.gamma)
    plt.show()
    df_['delta'] = [delta(df_.spot.iloc[i], df_.spot.iloc[i],
                          df_.time_before_expiration.iloc[i],
                          df_.volatility.iloc[i], 'CALL')
                    for i in range(len(df_))]
    plt.plot(df_.delta)
    plt.show()
    return df_


def get_gamma_from_our_options(df_):
    options_df = pd.read_csv('vp4_part.csv')
    # options_df = pd.read_csv('vp5_part.csv')
    options_df['date'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                          for d in options_df['date']]
    options_df = options_df[options_df.date.dt.hour == 8]
    options_df.date = options_df.date.dt.date
    # options_df.set_index('date', inplace=True)
    df_['gamma2'] = np.nan
    df_['options_count'] = np.nan
    df_['new_price'] = np.nan
    df_['extra_money_from_price'] = np.nan
    for i in range(len(df_)):
        todays_date = df_.date.iloc[i].date()
        print(todays_date)
        todays_gamma = options_df[options_df.date == todays_date].gamma
        real_gamma = df_.gamma.iloc[i]
        print(todays_gamma)
        df_.gamma2.iloc[i] = todays_gamma
        df_.options_count.iloc[i] = todays_gamma / real_gamma  # Количество опционов, которое должно быть на каждом шаге
        df_.new_price.iloc[i] = df_.options_count.iloc[i] * df_.price.iloc[i]
        if i > 0:
            options_we_need_to_buy = df_.options_count.iloc[i] - df_.options_count.iloc[i - 1]
            if options_we_need_to_buy > 0:  # Потратили деньгу, чтобы купить кусочек опциона
                df_.extra_money_from_price.iloc[i] = - df_.price.iloc[i] * options_we_need_to_buy
            else:  # Получили деньгу, потому что продали кусочек опциона
                df_.extra_money_from_price.iloc[i] = df_.price.iloc[i] * options_we_need_to_buy
    # print(options_df)
    print()
    print(df_)
    return df_


def add_money_diff(df_):
    df_['payout'] = np.nan
    prev_friday_idx = 5
    friday_count = 1
    for i in range(7, len(df_)):
        if df_.weekday.iloc[i] == 'Friday':
            friday_count += 1
            print(df_.date.iloc[i] - df_.date.iloc[prev_friday_idx], df_.date.iloc[i], df_.date.iloc[prev_friday_idx])

            if (df_.date.iloc[i] - df_.date.iloc[prev_friday_idx]).total_seconds() < 7 * 24 * 60 * 60 + 3 * 60 * 60:
                print('OK')
                print()
                df_.payout.iloc[i] = (df_.spot.iloc[i] - df_.spot.iloc[prev_friday_idx]) * df_.options_count.iloc[i]
            prev_friday_idx = i
    print('Friday count', friday_count)
    return df_


if __name__ == '__main__':
    df = pd.DataFrame()

    df = get_data_from_files(df, 'C:\\Users\\admin\\for python\\HISTORY\\options\\BTC\\old_version\\', 'old')
    df = get_data_from_files(df, 'C:\\Users\\admin\\for python\\HISTORY\\options\\BTC\\new_version\\', 'new')
    df = df.sort_values(by=['date'])
    df = add_delta_gamma(df)
    df = get_gamma_from_our_options(df)
    df = add_money_diff(df)
    print(df)
    # df.to_csv('deribit_part_vp5.csv', index=False)
    df.to_csv('deribit_part_vp4.csv', index=False)
