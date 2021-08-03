import csv
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import warnings
import datetime
import copy
from scipy import stats
import pwlf
# import time
from sklearn.linear_model import LinearRegression
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# warnings.filterwarnings("ignore")
# pd.options.mode.chained_assignment = None
from lib import plotter
from lib.exchange_data_reader_historical_new_format_backup import HistoricalReaderNewFormat
# from lib.exchange_data_reader_historical import HistoricalReader
from lib.last_table_reader import LastTableReader
# from lib.forward_price_counter import ForwardPriceCounter
# from lib.volatility_surface import Volatility
from lib.option import Option

from lib.useful_things import *
from lib.surface_creation import surface_object_from_file, surface_to_file
import lib.option_formulas as opt
from lib.option import get_years_before_expiration
import lib.plotter as pl
from lib.volatility import get_strike_by_delta, delta_slice_for_new_time, interpolate_surface
from lib.surface_creation import get_data_by_reader, get_surface


def get_strike_for_one_request_time(sabr, time):
    new_delta_c = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_c'],
                                           time)
    new_delta_p = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_p'],
                                           time)

    strike_c = get_strike_by_delta(sabr.strike_prices, new_delta_c, 0.25)
    strike_p = get_strike_by_delta(sabr.strike_prices, new_delta_p, -0.25)
    return strike_c, strike_p


def get_vol_from_surface(option, reader, today):
    index = np.where(np.array(reader.today) == today)[0][0]
    data = get_data_by_reader(reader, index)

    vol, points = get_surface(data)
    # print(f'MAXVOL {np.max(vol.surface)}')
    # pl.plot_surface(vol, points)
    # surface_to_file(vol, 'vol' + str(today)[:10] + '_' + str(today)[11:13] + '.csv')

    vol_short = interpolate_surface(vol.surface,
                                    vol.times_before_expiration,
                                    vol.strike_prices,
                                    option.time_before_expiration,
                                    option.strike_price)
    return vol_short


def make_vols_for_last_table(filename, asset):
    call_put_diff = 0.1
    N1, N2 = 0, 10000
    # read_row, step = 'first', 1
    read_row, step = 'last', 1
    # read_row, step = 'last', 60

    cut = True
    use_weights = True
    # asset = 'BTC'

    reader_btc = LastTableReader(filename)

    # reader_btc = HistoricalReader(file_path + 'OptBestBA_1m_Deribit_BTC_USDT_20210527.csv')
    reader_btc.get_data_from_file(call_put_diff, N1, N2, step)

    reader = reader_btc

    vol = np.full((5, 7, len(reader.today)), np.nan)
    delta = [0.1, 0.25, 0.5, 0.75, 0.9]
    df_spot = pd.DataFrame({'datetime': reader.today, 'spot': reader.spot})
    option_type = 'CALL'
    for k in range(len(reader.today)):
        today = reader.today[k]
        print('-' * 30)
        print(f'{k}: today: {[today, reader.today[k]]}')

        exp1w = today + timedelta(weeks=1)
        exp2w = today + timedelta(weeks=2)
        exp1m = today + timedelta(days=30)
        exp2m = today + timedelta(days=30 * 2)
        exp3m = today + timedelta(days=30 * 3)
        exp6m = today + timedelta(days=30 * 6)
        exp9m = today + timedelta(days=30 * 9)

        expiration_list = [exp1w, exp2w, exp1m, exp2m, exp3m, exp6m, exp9m]
        time_list = [get_years_before_expiration(today, expiration) for expiration in expiration_list]

        data = get_data_by_reader(reader, k)
        surface_obj, iv_list = get_surface(data, cut, use_weights)

        # for i, option_type in enumerate(['CALL']):
        for j in range(len(delta)):
            for m, time in enumerate(time_list):
                vol[j][m][k] = surface_obj.get_vol_by_time_delta(time, delta[j], option_type)

    # for i, option_type in enumerate(['CALL']):
    # for j, delta in enumerate([0.1, 0.25, 0.5]):
    for j in range(len(delta)):
        d = {'Datetime': reader.today}
        for m, exp_name in enumerate(['1w', '2w', '1m', '2m', '3m', '6m', '9m']):
            d[exp_name] = vol[j][m]

        df = pd.DataFrame(d)
        try:
            pd.read_csv(f'./every_6_hours_vol/1vol_d{delta[j] * 100:.0f}_{option_type}_{asset}.csv')
            df.to_csv(f'./every_6_hours_vol/1vol_d{delta[j] * 100:.0f}_{option_type}_{asset}.csv', mode='a', index=False,
                      header=False)
        except FileNotFoundError:
            df.to_csv(f'./every_6_hours_vol/1vol_d{delta[j] * 100:.0f}_{option_type}_{asset}.csv', index=False)

    try:
        pd.read_csv(f'./every_6_hours_spot/1spot_{asset}.csv')
        df_spot.to_csv(f'./every_6_hours_spot/1spot_{asset}.csv', mode='a', index=False, header=False)
    except FileNotFoundError:
        df_spot.to_csv(f'./every_6_hours_spot/1spot_{asset}.csv', index=False)


if __name__ == '__main__':
    make_vols_for_last_table("")
