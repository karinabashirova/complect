from lib.last_table_reader import LastTableReader
from lib.option_formulas import price_by_BS, OptionType, delta
from lib.surface_creation import get_data_by_reader, get_surface
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, Bounds, least_squares
from lib.exchange_data_reader_historical_new_format_backup import HistoricalReaderNewFormat
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from lib import plotter
import argparse
from kind_of_u_call import hestonMC, classicMC
import time


def strike_from_delta(S, T, sigma, delta, otype):
    if otype == "CALL":
        return S * np.exp(-norm.ppf(delta) * sigma * np.sqrt(T) + ((sigma ** 2) / 2) * T)
    else:
        return S * np.exp(norm.ppf(delta) * sigma * np.sqrt(T) + ((sigma ** 2) / 2) * T)


def get_vol():
    df = pd.read_csv('C:\\Users\\admin\\for python\\HISTORY\\spot\\shp-ohlcv_ftx_BTC-USDT_h.csv')
    return np.log(df[df.columns[1]] / df[df.columns[1]].shift(1)).rolling(14 * 24).std() * np.sqrt(365 * 24), \
           df[df.columns[1]][-2], df[df.columns[1]][-1]


def objective_function(params, real_spot, start_spot, variance_call, variance_put, kappa, theta, strike_call,
                       strike_put, real_BS_price):
    print(params, end=' --- ')

    Heston_intrinsic_call = hestonMC(spot=start_spot, time=1 / 365, variance=params[2],
                                     rho=params[0], r=0, kappa=kappa, lmbda=0, theta=theta, sigma=params[1],
                                     call_strike=strike_call, put_strike=strike_put,
                                     N=24, M=1000)

    # Heston_intrinsic_put = hestonMC(spot=start_spot, time=1 / 365, variance=params[2],
    #                                 rho=params[0], r=0, kappa=kappa, lmbda=0, theta=theta, sigma=params[1],
    #                                 call_strike=strike_call, put_strike=strike_put,
    #                                 N=24, M=1000)

    diff = real_BS_price - Heston_intrinsic_call  # , real_BS_price[1] - Heston_intrinsic_put[1]]
    er = np.sqrt(np.nansum(np.array(diff) ** 2))

    print(
        f'Heston {np.round([Heston_intrinsic_call])}'  # [0], Heston_intrinsic_put[1]], 2)}, '
        f'BS {np.round(real_BS_price, 2)} --- error {er:.4f}')

    return er


def main():
    df = pd.read_csv('C:\\Users\\admin\\for python\\HISTORY\\spot\\shp-ohlcv_ftx_BTC-USDT_h.csv')[::24]
    spot_list = df[df.columns[1]].values
    start_spot, real_spot = df[df.columns[1]].values[-2], df[df.columns[1]].values[-1]
    spot_list = spot_list[:-1]
    df = df.iloc[:-1]

    print(f'start spot: {start_spot}, real spot: {real_spot}')
    print(f'start spot date: {df[df.columns[0]].values[-2]}, real spot date: {df[df.columns[0]].values[-1]}\n')

    spot_variance = df[df.columns[1]].rolling(14).var() / df[df.columns[1]] ** 2
    variance_mean = spot_variance.rolling(14).mean()
    vol = np.log(df[df.columns[1]] / df[df.columns[1]].shift(1)).rolling(14).std() * np.sqrt(365)
    max_vol = np.nanmax(vol)
    min_vol = np.nanmin(vol)

    time_before_expiration = 1 / 365

    call_delta = 0.45
    put_delta = call_delta - 1

    call_strike = strike_from_delta(real_spot, time_before_expiration, max_vol, call_delta, 'CALL')
    put_strike = strike_from_delta(real_spot, time_before_expiration, min_vol, call_delta, 'PUT')

    print(f'call strike: {call_strike}, put strike: {put_strike}')
    print(f'min_vol: {min_vol}, max_vol: {max_vol}\n')

    sigma = np.log(vol / vol.shift(1)).std()  # * np.sqrt(365)
    rho = np.corrcoef(spot_list[14:], vol[14:])[0][1]
    variance = spot_variance.iloc[-1]
    theta = variance_mean.iloc[-1]

    print(f'Real sigma: {sigma}, real rho: {rho}\n')

    real_BS_price = np.array([price_by_BS(real_spot, call_strike, time_before_expiration, max_vol, 'CALL'),
                              price_by_BS(real_spot, put_strike, time_before_expiration, min_vol, 'PUT')])
    print('BS price', np.round(real_BS_price, 3))

    Heston_price_without_fitting = hestonMC(spot=start_spot, time=time_before_expiration, variance=max_vol ** 2,
                                            rho=rho, r=0, kappa=1, lmbda=0, theta=theta, sigma=sigma,
                                            call_strike=call_strike, put_strike=put_strike, N=24, M=1000)
    print('Heston no fitting', np.round(Heston_price_without_fitting, 3))
    print()

    bounds = ([-1, 1], [0, 5], [-np.inf, np.inf])
    start_params = np.array([0, 0, 0.5 * (max_vol ** 2 + min_vol ** 2)])
    start_time = time.time()
    res = minimize(objective_function, start_params, method='L-BFGS-B', bounds=bounds, options={'ftol': 0.01},
                   args=(
                       real_spot, start_spot, max_vol ** 2, min_vol ** 2, 1, theta, call_strike, put_strike,
                       real_BS_price))
    print('Time', time.time() - start_time)
    print(res)

    # rho_list = [-1, -0.8]#, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    # sigma_list = [1, 2, 3]#[0, 0.2, 0.4, 0.6, 0.8, 1]
    # rho_list_call, rho_list_put = np.full((len(rho_list), len(sigma_list)), np.nan), \
    #                               np.full((len(rho_list), len(sigma_list)), np.nan)
    #
    # for i, rho in enumerate(rho_list):
    #     for j, sigma in enumerate(sigma_list):
    #         print(rho, sigma)
    #         res = hestonMC(spot=start_spot, time=time_before_expiration, variance=variance,
    #                        rho=rho, r=0, kappa=1, lmbda=0, theta=theta, sigma=sigma,
    #                        call_strike=call_strike, put_strike=put_strike, N=24, M=1000)
    #         rho_list_call[i][j] = res[0]
    #         rho_list_put[i][j] = res[1]
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Surface(y=rho_list,
    #                          x=sigma_list,
    #                          z=rho_list_call, name='call',
    #                          opacity=0.75, colorscale='Viridis', showscale=False))
    #
    # fig.update_layout(
    #     title={
    #         'text': 'call',
    #         'y': 0.95,
    #         'x': 0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'
    #     }
    # )
    # fig.update_layout(
    #     scene=dict(
    #         yaxis=dict(
    #             title='rho'
    #         ),
    #         xaxis=dict(
    #             title='sigma')
    #     )
    # )
    # fig.show()
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Surface(y=rho_list,
    #                          x=sigma_list,
    #                          z=rho_list_put, name='put',
    #                          opacity=0.75, colorscale='Viridis', showscale=False))
    # fig.update_layout(
    #     title={
    #         'text': 'put',
    #         'y': 0.95,
    #         'x': 0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'
    #     }
    # )
    #
    # fig.update_layout(
    #     scene=dict(
    #         yaxis=dict(
    #             title='rho'
    #         ),
    #         xaxis=dict(
    #             title='sigma')
    #     )
    # )
    # fig.show()

    # plt.plot(sigma_list, rho_list_call, label='call rho')
    # plt.plot(sigma_list, [Heston_price_without_fitting[0]] * len(sigma_list), label='call no fit')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(sigma_list, rho_list_put, label='put rho')
    # plt.plot(sigma_list, [Heston_price_without_fitting[1]] * len(sigma_list), label='put no fit')  #
    # plt.legend()
    # plt.show()
    prices_by_heston_call = hestonMC(spot=start_spot, time=time_before_expiration, variance=res.x[2],  # max_vol ** 2,
                                     rho=res.x[0], r=0, kappa=1, lmbda=0, theta=theta, sigma=res.x[1],
                                     call_strike=call_strike, put_strike=put_strike, N=24, M=1000)[0]

    prices_by_heston_put = hestonMC(spot=start_spot, time=time_before_expiration, variance=res.x[2],  # max_vol ** 2,
                                    rho=res.x[0], r=0, kappa=1, lmbda=0, theta=theta, sigma=res.x[1],
                                    call_strike=call_strike, put_strike=put_strike, N=24, M=1000)[1]
    print('-' * 100)
    print('BS price', real_BS_price,
          '\nHeston fitting', prices_by_heston_call, prices_by_heston_put,
          '\nHeston without fitting', Heston_price_without_fitting)

    # print('-'*100)
    # call_strike = strike_from_delta(real_spot, time_before_expiration, max_vol, call_delta, 'CALL')+100
    # put_strike = strike_from_delta(real_spot, time_before_expiration, min_vol, call_delta, 'PUT')+500
    #
    # real_BS_price = np.array([price_by_BS(real_spot, call_strike, time_before_expiration, max_vol, 'CALL'),
    #                           price_by_BS(real_spot, put_strike, time_before_expiration, min_vol, 'PUT')])
    # print('BS price', np.round(real_BS_price, 3))
    # prices_by_heston = hestonMC(spot=start_spot, time=time_before_expiration, variance=max_vol**2,
    #                             rho=res.x[0], r=0, kappa=1, lmbda=0, theta=theta, sigma=res.x[1],
    #                             # rho=-0.90980793, r=0, kappa=1, lmbda=0, theta=theta, sigma=1.99691586,
    #                             call_strike=call_strike, put_strike=put_strike, N=24, M=1000)
    #
    # print('Heston price', prices_by_heston)


def main2():
    spot = 100.
    call_strike = 100
    put_strike = 100
    r = 0.05
    t = 30 / 365
    vol = 0.1

    print(price_by_BS(spot, call_strike, t, vol, 'CALL', r=r))
    print(price_by_BS(spot, put_strike, t, vol, 'PUT', r=r))

    a = classicMC(spot=spot, time=t, volatility=vol,
                  call_strike=call_strike, put_strike=put_strike, r=r, N=30, M=5000)
    print(a)

    a = hestonMC(spot=100., time=30 / 365, variance=0.01,
                 rho=-0.7, r=0.05, kappa=2, lmbda=0.05, theta=0.01, sigma=0.1,
                 call_strike=100., put_strike=100., N=30, M=5000)
    print(a)
    print(a[0] / (1 + r) ** (t), a[1] / (1 + r) ** (t))

    # print(price_by_BS(b[0], 100, 30/365, 0.01, 'CALL'),)


if __name__ == '__main__':
    main()
