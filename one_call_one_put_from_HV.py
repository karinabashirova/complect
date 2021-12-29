import csv
import sys
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import glob as gb

from lib.exchange_data_reader_historical_new_format_backup import HistoricalReaderNewFormat
from lib.exchange_data_reader_historical import HistoricalReader
# from lib.forward_price_counter import ForwardPriceCounter
# from lib.volatility_surface import Volatility
from lib.option import Option
from lib.useful_things import *
from lib.surface_creation import surface_object_from_file, surface_to_file
import lib.option_formulas as opt
from lib.option import get_years_before_expiration
import lib.plotter as pl
from lib.volatility import get_strike_by_delta, delta_slice_for_new_time, interpolate_surface
from lib.surface_creation import get_data_by_reader, get_surface, get_data_by_small_reader
from lib.option_formulas import price_by_BS
from lib.exchange_data_reader import ExchangeDataReader
from lib.small_reader import ReaderCutByDates


def count_RV_for_each_asset(filename_btc, first_vol_date, last_vol_date, days=30, cut_dates=True):
    df = pd.read_csv(filename_btc, usecols=[1, 2], header=None)

    time_ending = '00:00'
    df.columns = ['date', 'Spot']

    # df = df[[df.date[i][-len(time_ending):] == time_ending for i in range(len(df))]]

    df['RV'] = (np.log(df.Spot / df.Spot.shift(1))).rolling(days * 24 * 60).std() * np.sqrt(365 * 24 * 60)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    # df_btc['returns_btc'] = np.log(df_btc.Spot_btc / df_btc.Spot_btc.shift(1))

    if cut_dates:
        # print(first_vol_date)
        first_vol_date = datetime.datetime.strptime(first_vol_date,
                                                    '%Y-%b-%d %H:%M:%S')  # - datetime.timedelta(days=days)
        # print(first_vol_date)
        last_vol_date = datetime.datetime.strptime(last_vol_date, '%Y-%b-%d %H:%M:%S')
        # print(last_vol_date)
        df.date = [datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S') for d in df.date]

        # print(df.date.iloc[])
        try:
            idx1 = df.index[df.date == first_vol_date]  # np.argwhere(df_dates == first_vol_date)
            idx2 = df.index[df.date == last_vol_date]  # np.argwhere(df_dates == first_vol_date)
        except:
            print('Not enough data for volatility counting. Should be one month before your start date.')
            sys.exit(1)
        # print(np.argwhere(df_dates == first_vol_date))
        # idx2 = np.argwhere(df_dates == last_vol_date)
        # print(idx1, idx2)
        # if idx2[
        try:
            df = df[idx1[0]:idx2[0] + 1]
        except:
            print('!')
            pass

    # print(df.date.values[0], df.date.values[days * 24], df.date.values[-1])
    # print(df)
    return df
    # return df.RV_btc[days * 24:], df.returns_btc[days * 24:], df.RV_eth[days * 24:], \
    #        df.returns_eth[days * 24:], df.Spot_btc[days * 24:], df.Spot_eth[days * 24:], df.date[days * 24:]


def count_prices_for_one_put_one_call_options(df, start_time, end_time):
    df.reset_index(inplace=True)
    start_time_before_expiration = 60
    df_for_all_minutes = pd.DataFrame()
    df_for_all_minutes['start_date'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['real_date'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['spot'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['time_before_expiration'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['minutes_before_expiration'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['call_price'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['put_price'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['sum_price'] = [np.nan] * ((len(df) - 60) * 60)
    df_for_all_minutes['RV'] = [np.nan] * ((len(df) - 60) * 60)
    # print(len(df), (len(df) - 60) * 60, len(df_for_all_minutes))
    for i in range((len(df) - 60)):
        for j in range(60):
            # print(i, j, i * 60 + j)
            start_time_before_expiration = 60 - j
            df_for_all_minutes['start_date'].iloc[i * 60 + j] = df.date.iloc[i]
            df_for_all_minutes['real_date'].iloc[i * 60 + j] = df.date.iloc[i + j]
            df_for_all_minutes['spot'].iloc[i * 60 + j] = df.Spot.iloc[i + j]
            df_for_all_minutes['RV'].iloc[i * 60 + j] = df['RV'].iloc[i + j]
            df_for_all_minutes['time_before_expiration'].iloc[i * 60 + j] = start_time_before_expiration / 60 / 24 / 365
            df_for_all_minutes['minutes_before_expiration'].iloc[i * 60 + j] = start_time_before_expiration

            df_for_all_minutes['call_price'].iloc[i * 60 + j] = price_by_BS(df_for_all_minutes['spot'].iloc[i * 60 + j],
                                                                            df_for_all_minutes['spot'].iloc[i * 60 + j],
                                                                            df_for_all_minutes[
                                                                                'time_before_expiration'].iloc[
                                                                                i * 60 + j],
                                                                            df_for_all_minutes['RV'].iloc[i * 60 + j],
                                                                            'CALL')
            df_for_all_minutes['put_price'].iloc[i * 60 + j] = price_by_BS(df_for_all_minutes['spot'].iloc[i * 60 + j],
                                                                           df_for_all_minutes['spot'].iloc[i * 60 + j],
                                                                           df_for_all_minutes[
                                                                               'time_before_expiration'].iloc[
                                                                               i * 60 + j],
                                                                           df_for_all_minutes['RV'].iloc[i * 60 + j],
                                                                           'PUT')
    df_for_all_minutes['sum_price'] = df_for_all_minutes.call_price + df_for_all_minutes.put_price
    # print(df_for_all_minutes)
    df_for_all_minutes.to_csv(
        f'one_call_one_put_prices\\call_put_prices_for_start_time_{start_time[:-9]}T{start_time[-8:-6]}' \
        + f'T{start_time[-5:-3]}_for_end_time_{end_time[:-9]}T{end_time[-8:-6]}T{end_time[-5:-3]}.csv',
        index=False)


def count_prices_for_one_put_one_call_options_for_one_hour_only(df, start_time, end_time):
    df.reset_index(inplace=True)
    start_time_before_expiration = 60

    df['time_before_expiration'] = np.nan
    df['minutes_before_expiration'] = np.nan
    df['call_price'] = np.nan
    df['put_price'] = np.nan
    df['sum_price'] = np.nan

    for i in range(len(df)):
        df['time_before_expiration'].iloc[i] = start_time_before_expiration / 60 / 24 / 365
        df['minutes_before_expiration'].iloc[i] = start_time_before_expiration

        df['call_price'].iloc[i] = price_by_BS(df['Spot'].iloc[i], df['Spot'].iloc[i],
                                               df['time_before_expiration'].iloc[i],
                                               df['RV'].iloc[i], 'CALL')
        df['put_price'].iloc[i] = price_by_BS(df['Spot'].iloc[i], df['Spot'].iloc[i],
                                              df['time_before_expiration'].iloc[i],
                                              df['RV'].iloc[i], 'PUT')
    df['sum_price'] = df.call_price + df.put_price
    # print(df_for_all_minutes)
    df.to_csv(
        f'one_call_one_put_prices\\call_put_prices_for_start_time_{start_time[:-9]}T{start_time[-8:-6]}' \
        + f'T{start_time[-5:-3]}_for_end_time_{end_time[:-9]}T{end_time[-8:-6]}T{end_time[-5:-3]}.csv',
        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call and put options prices')

    parser.add_argument('start', type=str, help='Start date (for e.g. "2021-Feb-24 08:00:00")')
    parser.add_argument('end', type=str, help='End date (for e.g. "2021-Feb-24 11:00:00")')
    parser.add_argument('filename', type=str, help='File name with historical data')
    args = parser.parse_args()

    df = count_RV_for_each_asset(args.filename, first_vol_date=args.start,
                                 last_vol_date=args.end, cut_dates=True)
    # df.to_csv('df_rv.csv', index=False)
    # df = pd.read_csv('df_rv.csv')
    # count_prices_for_one_put_one_call_options(df, args.start, args.end)
    count_prices_for_one_put_one_call_options_for_one_hour_only(df, args.start, args.end)
