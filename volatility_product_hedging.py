import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib import option_formulas

from volatility_product_by_historical_data import get_historical_data, get_v1_v2_values, get_delta_v2_value


def cut_by_z_score(value, array, max_z_score):
    mean = np.mean(array)
    std = np.std(array)

    if abs(value - mean) / std > max_z_score:
        return np.sign(value - mean) * max_z_score * std + mean
    else:
        return value


def calculate_volatility(spot, days_for_volatility, max_z_score=None):
    z_score_window = 24 * 60

    spot_series = pd.Series(spot)

    returns = np.log(spot_series / spot_series.shift().values)

    if max_z_score is not None:
        for i in range(z_score_window, len(returns)):
            returns[i] = cut_by_z_score(returns[i], returns[i - z_score_window:i], max_z_score)

    volatility_window = days_for_volatility * 24 * 60
    historical_volatilities = pd.Series(returns).rolling(volatility_window).std() * np.sqrt(60 * 24 * 365)

    return historical_volatilities.values


def get_delta(spot, volatility):
    T = 1 / 24 / 365

    delta_call = np.full(len(spot), np.nan)
    delta_put = np.full(len(spot), np.nan)
    expiration = np.full(len(spot), np.nan)
    strikes = np.full(len(spot), np.nan)

    strike = spot[0]
    for i in range(0, len(spot)):
        minutes = i % 60
        if minutes == 0:
            strike = spot[i]

        expiration[i] = T - minutes / 60 / 24 / 365
        strikes[i] = strike

        delta_call[i] = option_formulas.delta(spot[i], strike, T - minutes / 60 / 24 / 365, volatility[i], 'CALL')
        delta_put[i] = option_formulas.delta(spot[i], strike, T - minutes / 60 / 24 / 365, volatility[i], 'PUT')

    df = pd.DataFrame(
        {'spot': spot, 'volatility': volatility, 'strike': strikes, 'exp': expiration, 'delta_call': delta_call,
         'delta_put': delta_put}
    ).to_csv('delta_hedging_v1.csv')

    return delta_call + delta_put


def get_hedging_part(spot, delta):
    hedging_part = np.full(len(spot[::60]), np.nan)

    for i in range(60, len(spot), 60):
        returns = np.array([np.log(spot[j] / spot[j - 1]) for j in range(i - 60, i)])
        # returns = np.array([spot[j] - spot[j - 1] for j in range(i - 60, i)])

        hedging_part[i // 60] = np.nansum(delta[i - 60:i] * returns)

    return hedging_part


def get_data():
    filename = 'BTCUSDT.csv'
    date = "2020-Aug-01 00:00:00"
    days = 365

    print('pd.read_csv')
    df_btc = pd.read_csv(filename, usecols=[1, 2], header=None, names=['date', 'Spot']).drop_duplicates('date')
    df_btc = df_btc.set_index('date').loc[date:].iloc[:days * 24 * 60 - 59]

    minutely_dates = np.array([datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S') for d in df_btc.index.values])
    minutely_spot = df_btc['Spot'].values

    volatility = calculate_volatility(minutely_spot, 1)

    print('get_historical_data')
    _, _, hourly_spot, hourly_dates = get_historical_data(filename, date, days)
    hourly_dates = np.array(hourly_dates)

    start_index1 = np.argwhere(~np.isnan(volatility))[0, 0]
    start_index2 = np.argwhere(hourly_dates == minutely_dates[start_index1])[0, 0]

    minutely_dates, minutely_spot, volatility = \
        minutely_dates[start_index1:], minutely_spot[start_index1:], volatility[start_index1:]
    hourly_dates, hourly_spot = \
        hourly_dates[start_index2:], hourly_spot[start_index2:]

    print(minutely_dates[::60][0], minutely_dates[::60][-1], len(minutely_dates[::60]), len(minutely_dates[::60]) / 24)
    print(hourly_dates[0], hourly_dates[-1], len(hourly_dates), len(hourly_dates) / 24)

    return hourly_spot, hourly_dates, minutely_spot, minutely_dates, volatility


def main_1_min_raw():
    hourly_spot, hourly_dates, minutely_spot, minutely_dates, minutely_volatility = get_data()

    print('get_delta')
    minutely_delta = get_delta(minutely_spot, minutely_volatility)

    print('get_hedging_part')
    hedging_part = get_hedging_part(minutely_spot, minutely_delta)

    print('get_delta_v2_value')
    v1 = get_delta_v2_value(hourly_spot, hourly_dates, 1000, 1000, 0.01, plot=False)

    plt.plot(np.nancumsum(hedging_part), label='cumulative delta-hedging part')
    plt.plot(v1, label='cumulative percent change of v1')
    plt.plot(np.nancumsum(hedging_part) + v1, alpha=0.5, label='sum of delta-hedging and v1 change')
    plt.legend()
    plt.grid()
    plt.show()


def main_n_min_from_file():
    df = pd.read_csv('delta_hedging_v1.csv')[1:]

    minutely_spot = df['spot'].astype(float).values
    minutely_delta = df['delta_call'].values + df['delta_put'].values

    step = 5
    d = minutely_delta[0]
    for i in range(len(minutely_delta)):
        if i % step == 0:
            d = minutely_delta[i]
        minutely_delta[i] = d

    hedging_part = get_hedging_part(minutely_spot, minutely_delta)

    v1 = pd.read_csv('delta_v.csv')['delta_v'].values[1:]

    plt.plot(np.nancumsum(hedging_part), label='cumulative delta-hedging part')
    plt.plot(np.nancumsum(v1), label='cumulative percent change of v1')
    plt.plot(np.nancumsum(hedging_part) + np.nancumsum(v1), label='sum of delta-hedging and v1 change')

    plt.title(f'step = {step}')
    plt.legend()
    plt.grid()
    plt.show()


def main_big_delta_change_from_file():
    df = pd.read_csv('delta_hedging_v1.csv')[1:]

    minutely_spot = df['spot'].astype(float).values
    minutely_delta = df['delta_call'].values + df['delta_put'].values

    max_diff = 0.1
    d = minutely_delta[0]
    count = 0
    for i in range(1, len(minutely_delta)):
        if abs(minutely_delta[i-1] - minutely_delta[i]) > max_diff:
            d = minutely_delta[i]
            count += 1
        minutely_delta[i] = d

    print(count/len(minutely_delta)*100)

    hedging_part = get_hedging_part(minutely_spot, minutely_delta)

    v1 = pd.read_csv('delta_v.csv')['delta_v'].values[1:]

    plt.plot(np.nancumsum(hedging_part), label='cumulative delta-hedging part')
    plt.plot(np.nancumsum(v1), label='cumulative percent change of v1')
    plt.plot(np.nancumsum(hedging_part) + np.nancumsum(v1), label='sum of delta-hedging and v1 change')

    plt.title(f'max_diff = {max_diff}')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # main_n_min_from_file()
    main_big_delta_change_from_file()
