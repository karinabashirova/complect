import datetime

import numpy as np
import pandas as pd

import argparse
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.stats import norm
from plotly.subplots import make_subplots
from scipy.optimize import minimize

from one_put_one_call_option_every_hour import get_reader_data_for_two_tables, \
    count_price_for_put_call_options_one_table
from volatility_product_by_historical_data import get_v1_v2_values_for_vols_pull


def count_vol_from_option_prices(spot, price):
    return np.sqrt(2 * np.pi / (1 / 24 / 365)) * price / spot


def count_historical_vol(spot):
    return np.log(spot[1] / spot[0]) * np.sqrt(365 * 24)


def count_v2_from_v1(v1_0, vol, percent):
    def func(v2_0):
        new_vol = (percent * v1_0 / v2_0[0])*np.sqrt(365*24)
        print('func', vol, new_vol, percent * v1_0 / v2_0[0])
        return np.abs(vol - new_vol)
    v2_0 = 100
    res = minimize(func, v2_0, method='L-BFGS-B')
    print(res)
    print('end', v1_0 / res.x[0] * percent)
    return res.x[0]


if __name__ == '__main__':
    percent = 0.01
    v1_0 = 100
    reader_list = get_reader_data_for_two_tables()
    call_price, put_price, vol1 = count_price_for_put_call_options_one_table(reader_list[0])

    S = [reader_list[0].spot, reader_list[1].spot]
    # v1, v2 = get_v1_v2_values_for_vols_pull(S, 100, 1000, 0.01)
    vol2 = count_vol_from_option_prices(S[0], call_price)
    # print(v1_0)
    v2_0 = count_v2_from_v1(v1_0, vol2, percent)
    v1, v2 = get_v1_v2_values_for_vols_pull(S, v1_0, v2_0, percent)
    # print(v1_0)
    # print(v1, v2, v2_0)
    h_vol = count_historical_vol(S)
    print(pd.DataFrame({'call price': [call_price],
                        'put price': [put_price],
                        'percent': [percent],
                        'value1_0': [v1_0],
                        'value2_0': [v2_0],
                        'value1': [v1],
                        'value2': [v2],
                        'prev_spot=prev_strike': [S[0]],
                        'next_spot': [S[1]],
                        'next_spot-prev_strike': [S[1] - S[0]],
                        'real_return': [np.log(S[1]/S[0])],
                        'volatility': [vol1]
                        }).transpose())

    print('V2 from equation')
    print(percent * v1_0 * np.sqrt(365*24)/vol1)

    x1 = (v1-100)/(1/24/365)
    x2 = (v2-1000)/(1/24/365)
    print((v1-100), x1, np.log(v1/100) * np.sqrt(365 * 24))
