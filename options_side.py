import csv
import sys
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
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
from lib.option_formulas import price_by_BS, vega, gamma, delta
from lib.exchange_data_reader import ExchangeDataReader
from lib.small_reader import ReaderCutByDates
from one_put_one_call_option_every_hour import get_list_of_filenames_to_use, get_vol_by_reader


def get_reader_data(start_date, end_date, path_to_files):
    dates, file_names, start_date, end_date = get_list_of_filenames_to_use(start_date, end_date, path_to_files)
    # print(dates)
    # print(file_names)
    reader_list = []
    for date, file in zip(dates, file_names):
        # print(f'Start of counting for {date}')
        cut_dates_reader = ReaderCutByDates(file)
        cut_dates_reader.read_one_csv()
        df_list = cut_dates_reader.cut_df(start_date, end_date)
        for df in df_list:
            # print(df)
            reader = ExchangeDataReader('00.csv', 0, 1000, df=df)
            reader.get_data_from_file()
            reader_list.append(reader)

    return reader_list


def get_weekdays(reader_list):
    weekdays = []
    for reader in reader_list:
        today = pd.to_datetime(reader.today, format='%Y-%m-%d %H:%M:%S')
        weekdays.append(today.weekday())
    return weekdays


def get_times_before_expirations(reader_list, weekdays):
    times = []
    todays = [pd.to_datetime(reader.today, format='%Y-%m-%d %H:%M:%S') for reader in reader_list]
    expirations = ()
    prev_i = 0
    for i, weekday in enumerate(weekdays):
        if weekday == 4 and todays[i].hour == 7:
            print(todays[i], prev_i, i)
            expirations += tuple([todays[i]] * len(todays[prev_i:i + 1]))
            prev_i = i + 1

    for i, reader in enumerate(reader_list):
        diff = expirations[i] - todays[i]
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        times.append(diff.total_seconds() / (365 * 24 * 60 * 60))

    return times, todays, expirations


def count_for_different_times_before_expiration(reader_list, times):
    vols_list = []
    delta_list = []
    gamma_list = []
    price_list = []
    for k, reader in enumerate(reader_list):
        # time_before_expiration =
        vol = get_vol_by_reader(reader, reader.spot, times[k])
        vols_list.append(vol)
        if not np.isnan(vol):
            delta_list.append(delta(reader.spot, reader.spot, times[k], vols_list[-1], 'CALL'))
            gamma_list.append(gamma(reader.spot, reader.spot, times[k], vols_list[-1]))
            price_list.append(
                price_by_BS(reader.spot, reader.spot, times[k], vol, 'CALL'))
        else:
            delta_list.append(np.nan)
            gamma_list.append(np.nan)
            price_list.append(np.nan)
    return vols_list, delta_list, gamma_list, price_list


def count_different_pnl(vols, deltas, prices, spot):
    deltas = np.array(deltas)
    prices = np.array(prices)
    spot = np.array(spot)
    delta_pnl = deltas[:-1] * (spot[1:] - spot[:-1])
    option_pnl = prices[1:] - prices[:-1]
    net_pnl = option_pnl - delta_pnl
    try:
        pd.DataFrame({'spot': spot, 'delta': deltas, 'prices': prices}).to_csv('options_side.csv')
    except:
        pass
    net_pnl = np.array(net_pnl)
    net_pnl = np.append(net_pnl, np.nan)
    return net_pnl


if __name__ == '__main__':
    start_date = "2021-05-28 07:59:51"
    end_date = "2021-06-11 07:57:45"
    path = "C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\folder_for_one_call_put"
    readers = get_reader_data(start_date, end_date, path)
    print(readers)
    weekdays = get_weekdays(readers)
    times_before_expirations, todays_list, expirations_list = get_times_before_expirations(readers, weekdays)
    spot_list = [reader.spot for reader in readers]
    vols_list, delta_list, gamma_list, price_list = count_for_different_times_before_expiration(readers,
                                                                                                times_before_expirations)
    pnl = count_different_pnl(vols_list, delta_list, price_list, spot_list)
    plt.plot(pnl)
    plt.show()
    pd.DataFrame({'date': todays_list, 'expiration': expirations_list,
                  'vol': vols_list, 'spot': spot_list,
                  'delta': delta_list, 'gamma': gamma_list,
                  'price': price_list, 'P&L': pnl}).to_csv('options_side1.csv')

    # for weekday, reader in zip(weekdays, readers):
    #     print('Today', reader.today, weekday)
