from datetime import timedelta
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from lib.last_table_reader import LastTableReader
from lib.option import get_years_before_expiration
from lib.surface_creation import get_data_by_reader, get_surface
from lib import plotter
from itertools import combinations

days = 30


def RV_IV_vs_IV_IV(filename):
    df = pd.read_csv(filename)
    df['RV'] = (np.log(df.Spot / df.Spot.shift(1))).rolling(days * 24).std() * np.sqrt(days * 24)

    X = np.log(df.RV[days * 24:].values / df['1m'][:-days * 24].values)
    Y = np.log(df['1m'][days * 24:].values / df['2m'][:-days * 24].values)
    hourly_y, hourly_x = [], []
    for i in range(24, len(X)):
        x = X[:i]
        y = Y[:i]
        z = np.polyfit(x, y, 1)
        hourly_y.append(z[0] * x[-1] + z[1])
        hourly_x.append(x[-1])
        # plt.plot(x, z[0] * x + z[1])
    Z = np.polyfit(X, Y, 1)
    print(Z)
    plt.scatter(X, Y)
    plt.plot(X, Z[0] * X + Z[1])
    X_sorted, hourly_res_sorted = zip(*sorted(zip(hourly_x, hourly_y), key=lambda k: k[0]))
    plt.plot(X_sorted, hourly_res_sorted)
    plt.show()
    print(len(X), len(Y), len(Z), len(df.Datetime[days * 24:]), len([np.nan] * 24 + hourly_y))
    pd.DataFrame({'Datetime': df.Datetime[days * 24:], 'X': X, 'Y': Y, 'Z': Z[0] * X + Z[1],
                  'hourly_z': [np.nan] * 24 + hourly_y}).to_csv(
        f'RV_IV_vs_IV_IV_{asset}.csv', index=False)


def RV_vs_return(filename):
    df = pd.read_csv(filename)
    df['RV'] = (np.log(df.Spot / df.Spot.shift(1))).rolling(days * 24).std() * np.sqrt(days * 24)
    df['returns'] = np.log(df.Spot.values / df.Spot.shift(days * 24).values)
    X = df.RV[days * 24:].values
    Y = df.returns[days * 24:].values

    hourly_y, hourly_x = [], []
    for i in range(24, len(X)):
        x = X[:i]
        y = Y[:i]
        z = np.polyfit(x, y, 1)
        hourly_y.append(z[0] * x[-1] + z[1])
        hourly_x.append(x[-1])
        # plt.plot(x, z[0] * x + z[1])

    Z = np.polyfit(X, Y, 1)
    print(Z)
    plt.scatter(X, Y)
    plt.plot(X, Z[0] * X + Z[1])
    X_sorted, hourly_res_sorted = zip(*sorted(zip(hourly_x, hourly_y), key=lambda k: k[0]))
    plt.plot(X_sorted, hourly_res_sorted)
    plt.show()
    pd.DataFrame({'Datetime': df.Datetime[days * 24:], 'X': X, 'Y': Y, 'Z': Z[0] * X + Z[1],
                  'hourly_z': [np.nan] * 24 + hourly_y}).to_csv(
        f'RV_vs_return_{asset}.csv', index=False)


def IV_vs_return(filename):
    df = pd.read_csv(filename)
    df['returns'] = np.log(df.Spot / df.Spot.shift(days * 24))
    X = np.log(df['1m'][days * 24:].values / df['2m'][:-days * 24].values)
    Y = df.returns[days * 24:].values
    hourly_y, hourly_x = [], []

    for i in range(24, len(X)):
        x = X[:i]
        y = Y[:i]
        z = np.polyfit(x, y, 1)
        hourly_y.append(z[0] * x[-1] + z[1])
        hourly_x.append(x[-1])
        # plt.plot(x, z[0] * x + z[1])

    Z = np.polyfit(X, Y, 1)
    print(Z)
    plt.scatter(X, Y)
    plt.plot(X, Z[0] * X + Z[1])
    X_sorted, hourly_res_sorted = zip(*sorted(zip(hourly_x, hourly_y), key=lambda k: k[0]))
    plt.plot(X_sorted, hourly_res_sorted)
    plt.show()
    pd.DataFrame({'Datetime': df.Datetime[days * 24:], 'X': X, 'Y': Y, 'Z': Z[0] * X + Z[1],
                  'hourly_z': [np.nan] * 24 + hourly_y}).to_csv(
        f'IV_vs_return_{asset}.csv', index=False)


def strange_thing_for_z_score(filename):
    df = pd.read_csv(filename)
    # df.set_index('Datetime', inplace=True)
    df_output = pd.DataFrame({'Datetime': df.Datetime})
    df.drop(columns=['Spot', 'Datetime', 'Real_datetime'], inplace=True)
    print(df)
    matrix = pd.DataFrame(columns=df.columns, index=df.columns)
    for c in combinations(df.columns, 2):
        exp1, exp2 = c

        iv1 = df[exp1].values
        iv2 = df[exp2].values

        zscore_iv1 = (np.log(iv1) - np.nanmean(np.log(iv1))) / np.nanstd(np.log(iv1))
        zscore_iv2 = (np.log(iv2) - np.nanmean(np.log(iv2))) / np.nanstd(np.log(iv2))

        beta = np.corrcoef(zscore_iv1, zscore_iv2)[0][1]
        if np.isnan(beta):
            check_list = [np.argwhere((~np.isnan(zscore_iv1)) & (~np.isnan(zscore_iv2))).flatten()]
            beta = np.corrcoef(zscore_iv1[check_list], zscore_iv2[check_list])[0][1]

        y = iv1 - beta * iv2
        zscore = (y - np.nanmean(y)) / np.nanstd(y, ddof=1) * np.sqrt(len(y))
        df_output[exp1 + '_' + exp2] = zscore
        matrix[exp1][exp2] = beta

    matrix.to_csv(f'corr_matrix_{asset}.csv')
    df_output.to_csv(f'z_scores_for_IV_{asset}.csv')
    # df_output['1m_3m'].plot()
    # plt.show()
    plt.scatter(iv1, iv2)
    plt.plot(iv1, beta * iv1, color='red')
    plt.show()


# asset = 'BTC'
for asset in ['BTC', 'ETH']:
    filename = f'./vols/no_max_cut_empty_dates_IV3_cut_weights_{asset}vol_d50_CALL.csv'
    RV_IV_vs_IV_IV(filename)
    RV_vs_return(filename)
    IV_vs_return(filename)
    strange_thing_for_z_score(filename)
