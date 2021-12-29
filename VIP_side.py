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


def get_data_from_options_side():
    df = pd.read_csv('options_side1.csv')
    return df


def count_v2_from_v1(v1_0, vol, percent):
    def func(v2_0):
        new_vol = (percent * v1_0 / v2_0[0]) * np.sqrt(365 * 24)
        # print('func', vol, new_vol, percent * v1_0 / v2_0[0])
        return np.abs(vol - new_vol)

    v2_0 = 100
    res = minimize(func, v2_0, method='L-BFGS-B')
    # print(res)
    # print('end', v1_0 / res.x[0] * percent)
    return res.x[0]


def count_ratio(percent, value1, df):
    df['value1'] = [value1] * len(df)
    print(1)
    df['value2_from_IV'] = [count_v2_from_v1(df.value1.iloc[i],
                                             df.vol.iloc[i],
                                             percent) for i in range(len(df))]
    df['from_V1'] = df.value1 * percent
    df['from_V2'] = np.nan

    df['return'] = np.nan
    df['return'] = df.spot / df.spot.shift(1) - 1

    df['from_V2'] = df.value2_from_IV * (df.spot / df.spot.shift(1) - 1)
    df['from_options'] = np.nan
    print(3)
    df.from_options = np.abs(df.spot / df.spot.shift(1) - 1)
    df['price_1hour_expiry'] = [price_by_BS(df.spot.iloc[i], df.spot.iloc[i], 1 / 24 / 365, df.vol.iloc[i], 'CALL')
                                for i in range(len(df))]
    df['gamma_1hour_expiry'] = [gamma(df.spot.iloc[i], df.spot.iloc[i], 1 / 24 / 365, df.vol.iloc[i])
                                for i in range(len(df))]

    return df


def count_pnl(df, percent, fixed_fee, X):
    df['P&L_VIP'] = (df.value1 * percent - df.value2 * fixed_fee) * X
    df.to_csv('total_pnl.csv')


if __name__ == '__main__':
    percent = 0.01
    v1_0 = 100
    df = get_data_from_options_side()
    new_df = count_ratio(percent, v1_0, df)
    new_df.to_csv('VIP_part.csv', index=False)
