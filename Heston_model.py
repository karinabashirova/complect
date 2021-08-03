import datetime
import sys
from lib.last_table_reader import LastTableReader
from lib.option_formulas import price_by_BS, OptionType, delta, strike_from_delta, cdf, pdf
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
# from kind_of_u_call import hestonMC, classicMC
import time
from lib.useful_things import pprint
# from numpy import *
# from sympy import Symbol, integrate, re, sqrt, exp, log, sinh, cosh
from scipy.integrate import quad, quadrature
from scipy.stats import shapiro
from icecream import ic


def get_heston_params_from_2_1d_options_and_count_prices_for_different_strikes_and_expirations():
    def objective_function(params, call_bs_price_for_1day_fit, put_bs_price_for_1day_fit):
        call_heston_price = price_by_heston(strike=call_strike, T=expiration_call,
                                            kappa=params[0], theta=params[1], sigma=params[2],
                                            rho=params[3], v0=params[4], S0=spot, r=r, option_type='CALL')

        put_heston_price = price_by_heston(strike=put_strike, T=expiration_call,
                                           kappa=params[0], theta=params[1], sigma=params[2],
                                           rho=params[3], v0=params[4], S0=spot, r=r, option_type='PUT')

        diff = np.array([call_bs_price_for_1day_fit - call_heston_price, put_bs_price_for_1day_fit - put_heston_price])
        return np.sqrt(np.sum(diff ** 2))

    spot = 100
    r = 0.01
    expiration_call = 1 / 365
    bounds = ([-20, 20], [1e-3, 20], [1e-3, 20], [-1, 1], [1e-3, 20])
    call_strikes = np.arange(spot, 125, 1)
    put_strikes = np.arange(75, spot + 1, 1)

    price_BS_call = np.full((len(call_strikes), len(put_strikes)), np.nan, dtype=float)
    price_BS_put = np.full((len(call_strikes), len(put_strikes)), np.nan, dtype=float)
    price_H_call = np.full((len(call_strikes), len(put_strikes)), np.nan, dtype=float)
    price_H_put = np.full((len(call_strikes), len(put_strikes)), np.nan, dtype=float)

    call_strike = call_strikes[-1]
    put_strike = put_strikes[-1]
    price_BS_call_1 = price_by_BS(S=spot, K=call_strike, T=expiration_call, sigma=0.6, option_type='CALL',
                                  r=r)
    price_BS_put_1 = price_by_BS(S=spot, K=put_strike, T=expiration_call, sigma=1, option_type='PUT', r=r)
    start_params = np.array([0, 0.20, 0.30, -0.60, 0.20])

    res = minimize(objective_function, start_params,
                   method='L-BFGS-B', args=(price_BS_call_1, price_BS_put_1),
                   bounds=bounds,  # options={'disp': True}
                   )
    params = res.x
    for i, call_strike in enumerate(call_strikes):

        for j, put_strike in enumerate(put_strikes):
            print(call_strike, put_strike)
            price_BS_call[i][j] = price_by_BS(S=spot, K=call_strike, T=expiration_call, sigma=0.6, option_type='CALL',
                                              r=r)
            price_BS_put[i][j] = price_by_BS(S=spot, K=put_strike, T=expiration_call, sigma=1, option_type='PUT', r=r)

            price_H_call[i][j] = price_by_heston(strike=call_strike, T=expiration_call,
                                                 kappa=params[0], theta=params[1], sigma=params[2],
                                                 rho=params[3], v0=params[4], S0=spot, r=r,
                                                 option_type='CALL')
            price_H_put[i][j] = price_by_heston(strike=put_strike, T=expiration_call,
                                                kappa=params[0], theta=params[1], sigma=params[2],
                                                rho=params[3], v0=params[4], S0=spot, r=r,
                                                option_type='PUT')
    print(price_BS_call)
    print(price_BS_put)
    print(price_H_call)
    print(price_H_put)


def plot_price(price_BS, price_H, name, strikes, times):
    fig = go.Figure()
    error_surface = np.abs(price_BS - price_H) / price_BS
    print(error_surface)

    error_surface[error_surface > 18] = np.nan
    print(error_surface)
    fig.add_trace(go.Surface(x=strikes,
                             y=times,
                             z=error_surface, name='BS' + name,
                             opacity=0.75, colorscale='Plotly3', showscale=False))

    # fig.add_trace(go.Surface(x=strikes,
    #                          y=times,
    #                          z=price_H, name='Heston' + name,
    #                          opacity=0.75, colorscale='Viridis', showscale=False))
    fig.update_layout(
        title={
            'text': name,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='strikes'
            ),
            yaxis=dict(
                title='times'),
            zaxis=dict(
                title='|BS - H|')
        )
    )
    fig.show()


def characteristic_function(u, t, kappa, theta, sigma, rho, v0, S0, r):
    ksi = kappa - sigma * rho * 1j * u
    d = np.sqrt(ksi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))
    D = np.log(d) + (kappa - d) * t / 2. - np.log((d + ksi) / 2. + (d - ksi) * np.exp(-d * t) / 2.)

    A1 = (u ** 2 + 1j * u) * np.sinh(d * t / 2)
    A2 = d * np.cosh(d * t / 2.) + ksi * np.sinh(d * t / 2.)
    A = A1 / A2

    phi = np.exp(1j * u * (np.log(S0) + r * t) - t * kappa * theta * rho * 1j * u / sigma -
                 v0 * A + 2 * kappa * theta / sigma ** 2 * D)
    return phi


def price_by_heston(strike, T, kappa, theta, sigma, rho, v0, S0, r, option_type):
    integarnd1 = lambda u: \
        np.real(np.exp(-1j * u * np.log(strike)) / (1j * u) *
                characteristic_function(u - 1j, T, kappa, theta, sigma, rho, v0, S0, r))

    integarnd2 = lambda u: \
        np.real(np.exp(-1j * u * np.log(strike)) / (1j * u) *
                characteristic_function(u, T, kappa, theta, sigma, rho, v0, S0, r))

    a = 0
    b = 2500

    int1 = quad(integarnd1, a, b)[0]
    int2 = quad(integarnd2, a, b)[0]

    while (np.isnan(int1) or np.isnan(int2)) and b > 100:
        b -= 100
        int1 = quad(integarnd1, a, b)[0]
        int2 = quad(integarnd2, a, b)[0]

    c = 0.5 * (S0 - np.exp(-r * T) * strike) + \
        np.exp(-r * T) / np.pi * (int1 - strike * int2)

    if option_type == 'CALL':
        return c
    else:
        return c - np.exp(-r * T) * S0 + np.exp(-r * T) * strike


def get_product_price(call_percent, put_percent, S, r, days=30):
    M, N = S.shape
    steps = N // days

    # M = 1000

    nan_count = np.zeros_like(S)

    intrinsic_call = np.zeros(days)
    intrinsic_put = 0

    tokens = np.zeros(days)

    for m in range(M):
        call_strike = call_percent * S[m][0]
        token_count = 1
        c = 0

        for n in range(N):  # steps+1
            if n % steps == 0 and n > 0:
                if np.isnan(S[m][n]):
                    nan_count[m][n] = 1

                c_from_prev_step = c
                c = np.exp(-1 / 365 * r) * max(0, S[m][n] - call_strike) * token_count

                intrinsic_call[n // steps - 1] += c

                if max(0, S[m][n] - call_strike) != 0:
                    token_count = call_strike * token_count / S[m][n]

                token_count += max(0, c_from_prev_step / S[m][n - steps])
                tokens[n // steps - 1] += token_count

                call_strike = call_percent * S[m][n]

        # c = np.exp(-1 / 365 * r) * max(0, S[m][-1] - call_strike) * token_count
        # intrinsic_call[-1] += c
        #
        # token_count += max(0, c / S[m][-1])
        # tokens[-1] += token_count

        p = np.exp(-days / 365 * r) * max(0, S[m][0] * put_percent - S[m][-1])
        intrinsic_put += p

    print(f'\nNaN: {100 * np.mean(nan_count):.2f}%')
    # print(f'NaN axis 0 mean shape: {np.mean(nan_count, axis=0).shape}')
    # print(f'NaN axis 1 mean shape: {np.mean(nan_count, axis=1).shape}')

    print('\nTOKENS')
    print(tokens / (M - np.mean(nan_count, axis=0)[steps::steps]))

    return intrinsic_call / (M - np.mean(nan_count, axis=0)[steps::steps]), intrinsic_put / (M - np.mean(nan_count))


def product_price(method, spot, r, call_percent, put_percent, volatility=None, params=None):
    if method == 'BS':
        if volatility is not None:
            S = spot_bs_mc(days=30, volatility=volatility, S0=spot, r=r, M=5000, steps=12)
        else:
            print('Volatility is None')
            sys.exit(2)
    else:
        if params is not None:
            S = spot_heston_mc(days=30, *params, S0=spot, r=r, M=1000, steps=12)
        else:
            print('Heston params vector is None')
            sys.exit(2)

    call_list, put = get_product_price(call_percent, put_percent, S, r=r)

    print(f'CALL: {call_list}\nCALL SUM: {np.sum(call_list)}\nPUT: {put}')


def price_by_heston_mc(call_strike, put_strike, days, S, r, steps=24):
    intrinsic_call = 0
    intrinsic_put = 0

    M = len(S)

    last_day_index = days * steps
    if last_day_index == len(S[0]):
        last_day_index = -1

    for m in range(M):
        intrinsic_call += np.exp(-days / 365 * r) * max(0, S[m][last_day_index] - call_strike)
        intrinsic_put += np.exp(-days / 365 * r) * max(0, put_strike - S[m][last_day_index])

    return intrinsic_call / M, intrinsic_put / M


def price_by_heston_approximate(strike, T, kappa, theta, sigma, rho, v0, S0, r, option_type='PUT'):
    spot_mult = np.sqrt(2 * v0 * T) / (4 * np.sqrt(np.pi)) - np.sqrt(2 * T) * rho * sigma / (4 * np.sqrt(np.pi * v0))

    strike_mult = np.sqrt(2 * v0 * T) / (4 * np.sqrt(np.pi)) + np.sqrt(2 * T) * rho * sigma / (4 * np.sqrt(np.pi * v0))

    exp_degree = -(S0 - strike) ** 2 / (2 * v0 * T * strike ** 2)

    last_add = -(S0 + (r * T - 1) * strike) * cdf((strike - S0) / (strike * np.sqrt(v0 * T)))

    p = (S0 * spot_mult + strike * strike_mult) * np.exp(exp_degree) + last_add
    if option_type == 'PUT':
        return p
    else:
        return p + np.exp(-r * T) * S0 - np.exp(-r * T) * strike


def delta_by_finite_difference(params, spot, call_strike, put_strike, days, r, price_by_spot):
    print(price_by_heston(call_strike, put_strike, days, *params, spot, r))
    ds = 1
    price_by_spot_plus = get_product_price(call_strike, put_strike, days, *params, spot + ds, r)
    price_by_spot = np.sum(price_by_spot[0]) + price_by_spot[1]
    price_by_spot_plus = np.sum(price_by_spot_plus[0]) + price_by_spot_plus[1]
    dv = price_by_spot_plus - price_by_spot

    return dv / ds


def spot_heston_mc(days, kappa, theta, sigma, rho, v0, S0, r, M=10000, steps=24 * 12):
    delta_t = 1 / 365 / steps

    N = days * steps + 1

    # S = np.full((M, N), S0, dtype=float)
    S = np.full(N, S0, dtype=float)
    print(f'SPOT CALCULATING --- {M} iterations')
    # for j in range(M):
    check_not_finish = True
    j = 0
    while check_not_finish:
        # print(j, end=' ')
        St = S0
        vt = v0

        lnSt = np.log(St)
        lnvt = np.log(vt)

        for n in range(1, N):  # steps+1
            e = norm.ppf(np.random.random())
            eS = norm.ppf(np.random.random())
            ev = rho * eS + np.sqrt(1 - rho ** 2) * e

            lnSt += (r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * eS
            lnvt += (1 / vt) * (kappa * (theta - vt) - 0.5 * sigma ** 2) * delta_t + \
                    sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev

            St = np.exp(lnSt)
            vt = np.exp(lnvt)

            S[n] = St
            j += 1
        if not np.isnan(np.sum(S)):
            check_not_finish = False
    print()
    return S


def spot_bs_mc(days, volatility, S0, r, M=1000, steps=24 * 12):
    print(f'SPOT CALCULATING --- {M} iterations')

    dt = 1 / 365 / steps

    N = days * steps

    S = np.full((M, N + 1), S0, dtype=float)

    for m in range(M):
        if m % 100 == 0:
            print(m, end=' ')

        if m > 0 and m % 1000 == 0 or m == M - 1:
            print()

        Z = np.random.normal(size=N)

        for n in range(N):
            S[m][n + 1] = S[m][n] * np.exp(
                (r - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z[n])

    return S


def get_heston__params_from_2bs_prices(spot, call_strike, put_strike, call_vol, put_vol, r,
                                       call_expiration=1 / 365, put_expiration=1 / 365,
                                       is_print=True):
    bs_call = price_by_BS(S=spot, K=call_strike, T=call_expiration, sigma=call_vol, option_type='CALL', r=r)
    bs_put = price_by_BS(S=spot, K=put_strike, T=put_expiration, sigma=put_vol, option_type='PUT', r=r)

    bounds = np.array([[-10, 10], [1e-3, 10], [1e-3, 10], [-1, 1], [1e-3, 10]])
    start_params = np.array([1.2, 0.20, 0.30, -0.60, 0.20])

    def objective_function(params):
        if is_print:
            print(f'kappa {params[0]:.4f}, theta {params[1]:.4f}, sigma {params[2]:.4f},',
                  f'rho {params[3]:.4f}, v0 {params[4]:.4f}',
                  end=' --- ')

        heston_call = price_by_heston(strike=call_strike, T=call_expiration,
                                      kappa=params[0], theta=params[1], sigma=params[2],
                                      rho=params[3], v0=params[4], S0=spot, r=r, option_type='CALL')

        heston_put = price_by_heston(strike=put_strike, T=put_expiration,
                                     kappa=params[0], theta=params[1], sigma=params[2],
                                     rho=params[3], v0=params[4], S0=spot, r=r, option_type='PUT')

        diff = np.array([bs_call - heston_call, bs_put - heston_put])

        er = np.sqrt(np.sum(diff ** 2))
        if is_print:
            print(f'{er:.6f}')
        return diff

    res = least_squares(objective_function, start_params, method='dogbox', bounds=bounds.T, xtol=0)

    print(res)

    return res.x


def hedging_product_hestonMC(call_strike, put_strike, days, S, r, steps=24):
    M, N = S.shape
    M = 13

    nan_count = np.zeros_like(S)
    hedge_result = np.zeros_like(S)

    intrinsic_call = np.zeros(days)
    intrinsic_put = 0

    tokens = np.zeros(days)
    ds = 0.1
    # call_strikes = []
    for m in range(12, 13):
        # call_strike = 1.05 * S[m][0]
        token_count = 1
        c = 0
        c_ds = 0
        p = 0
        p_ds = 0

        for n in range(N):  # steps+1
            c_from_prev_step = c
            time_call = 1 - (n % 24) / 24
            c = np.exp(-time_call / 365 * r) * max(0, S[m][n] - call_strike) * token_count
            c_ds = np.exp(-time_call / 365 * r) * max(0, S[m][n] - ds - call_strike) * token_count
            delta_call_ds = (c - c_ds) / ds

            time_put = 30 - (n % 24) / 24
            p_from_prev_step = p
            p = np.exp(-time_put / 365 * r) * max(0, -S[m][n] + call_strike)
            p_ds = np.exp(-time_put / 365 * r) * max(0, -S[m][n] + ds - call_strike)
            delta_put_ds = (p - p_ds) / ds

            hedging_part = (delta_call_ds + delta_put_ds) * (S[m][n] - S[m][n - 1])
            option_part = (c - c_from_prev_step) + (p - p_from_prev_step)
            hedge_result[m][n] = hedging_part - option_part
            if np.round(hedging_part, 3) != np.round(option_part, 3):
                print(n, hedging_part, option_part, delta_call_ds, delta_put_ds)
                print(S[m][n], S[m][n - 1], c, c_from_prev_step, p, p_from_prev_step)
                print()

            if n % steps == 0 and n > 0:
                if np.isnan(S[m][n]):
                    nan_count[m][n] = 1

                intrinsic_call[n // steps - 1] += c

                if max(0, S[m][n] - call_strike) != 0:
                    token_count = call_strike * token_count / S[m][n]

                token_count += max(0, c_from_prev_step / S[m][n - steps])
                tokens[n // steps - 1] += token_count

                # call_strike = 1.05 * S[m][n]

        c_from_prev_step = c
        c = np.exp(-1 / 24 / 365 * r) * max(0, S[m][-1] - call_strike) * token_count
        intrinsic_call[-1] += c

        c_ds = np.exp(-1 / 24 / 365 * r) * max(0, S[m][-1] - ds - call_strike) * token_count
        delta_call_ds = (c - c_ds) / ds

        token_count += max(0, c / S[m][-1])
        tokens[-1] += token_count
        p_from_prev_step = p
        p = np.exp(-days / 365 * r) * max(0, put_strike - S[m][-1])
        intrinsic_put += p

        p_ds = np.exp(-1 / 24 / 365 * r) * max(0, -S[m][-1] + ds - call_strike)
        delta_put_ds = (p - p_ds) / ds
        hedging_part = (delta_call_ds + delta_put_ds) * (S[m][-1] - S[m][-1 - 1])
        option_part = (c - c_from_prev_step) + (p - p_from_prev_step)
        hedge_result[m][-1] = hedging_part - option_part
        print(n, hedging_part, option_part)

    print(f'NaN: {100 * np.mean(nan_count):.2f}%')
    print(f'NaN axis 0 mean shape: {np.mean(nan_count, axis=0).shape}')
    print(f'NaN axis 1 mean shape: {np.mean(nan_count, axis=1).shape}')

    print('Hedging results, axes = 0', np.nanmean(hedge_result, axis=0).shape)
    print('Hedging results, axes = 0', np.nanmean(hedge_result, axis=1).shape)
    plt.plot(np.nanmean(hedge_result, axis=0))
    plt.show()
    # plt.plot(S[0])
    # plt.show()
    # print('\nTOKENS')
    # print(tokens / (M - np.append(np.mean(nan_count, axis=0)[24::24], np.mean(nan_count, axis=0)[-1])))

    return intrinsic_call / (M - np.append(np.mean(nan_count, axis=0)[24::24], np.mean(nan_count, axis=0)[-1])), \
           intrinsic_put / (M - np.mean(nan_count)), S[12]


def hedging_product_heston_df(S, r, params, call_expiration=1 / 365, put_expiration=30 / 365):
    S = np.array(S)
    df = pd.DataFrame({'Spot': S})
    strikes_call = np.array([1.01 * S[:-1:24]] * 24).T.flatten()
    df['Strikes'] = np.append(strikes_call, np.nan)
    put_strike = 0.9 * S[0]

    print(price_by_heston(strikes_call[0], 1 / 365, *params, S[0], r, 'CALL'))

    ds = 0.1
    price_call_S_plus_ds = np.array(
        [price_by_heston(call_strike, call_expiration - (i % 24) / 24 / 365, *params, S[i] + ds, r, 'CALL')
         for i, call_strike in enumerate(strikes_call)])
    price_call_S_minus_ds = np.array(
        [price_by_heston(call_strike, call_expiration - (i % 24) / 24 / 365, *params, S[i] - ds, r, 'CALL')
         for i, call_strike in enumerate(strikes_call)])
    price_call_S = np.array(
        [price_by_heston(call_strike, call_expiration - (i % 24) / 24 / 365, *params, S[i], r, 'CALL')
         for i, call_strike in enumerate(strikes_call)])

    price_put_S_plus_ds = np.array(
        [price_by_heston(put_strike, put_expiration - i / 24 / 365, *params, S[i] + ds, r, 'PUT')
         for i in range(len(S) - 1)])
    price_put_S_minus_ds = np.array(
        [price_by_heston(put_strike, put_expiration - i / 24 / 365, *params, S[i] - ds, r, 'PUT')
         for i in range(len(S) - 1)])
    price_put_S = price_by_heston(put_strike, put_expiration, *params, S[0], r, 'PUT')
    df['Price_of_put_options_plus_ds'] = np.append(price_put_S_plus_ds, np.nan)
    df['Price_of_put_options_minus_ds'] = np.append(price_put_S_minus_ds, np.nan)
    # price_put_S = np.array([price_by_heston(put_strike, put_expiration - i / 24 / 365, *params, S[i], r, 'PUT')
    #                         for i in range(len(S))])
    print('Put option price', price_put_S)
    print('30 Call options price', np.sum(price_call_S[::24]))

    intrinsic_put = max(0, put_strike - S[-1])
    delta_call = (price_call_S_plus_ds - price_call_S_minus_ds) / 2 / ds
    delta_put = (price_put_S_plus_ds - price_put_S_minus_ds) / 2 / ds
    delta_ds = delta_call - delta_put

    R = np.array([delta_ds[i - 1] * (S[i] - S[i - 1]) for i in range(1, len(S))])

    P = np.array([max(0, S[i * 24] - strikes_call[(i - 1) * 24]) for i in range(1, 30)])
    P = np.append(P, max(0, S[-1] - strikes_call[-2]))

    Product_price = np.sum(price_call_S[::24]) - price_put_S  # стартовая цена продукта

    strange_total_thing = (Product_price + np.sum(R) - (np.sum(P) - intrinsic_put)) / np.sum(P)
    print('Total check', strange_total_thing)
    print()
    print('R_m', Product_price + np.sum(R))
    print('Sum(R_i)', np.sum(R))
    print('Sum(P_i)', np.sum(P))
    print('Product_price', Product_price)
    print('Intrinsic put', intrinsic_put)


def get_mc_spot_for_one_iteration(S0, volatility, T=30 / 365, r=0., N=30 * 24):
    S = np.full(N + 1, S0, dtype=float)

    delta_t = T / N
    # delta_t_0 = T / N
    Z = np.random.normal(size=N)
    for n in range(N):
        # print(n, delta_t)
        S[n + 1] = S[n] * np.exp(
            (r - 0.5 * volatility ** 2) * delta_t + volatility * np.sqrt(delta_t) * Z[n])
        # delta_t += delta_t_0

    return S


def hedging_HESTON(S, r, params, call_expiration=1 / 365, put_expiration=30 / 365):
    call_strike = 29000.0
    put_strike = 20000
    volatility0 = 0.26758712

    S0 = 31893.78

    steps = 24 * 12
    days = 30
    T = days / 365
    N = days * steps
    # r = 0.0

    np.random.seed(42)

    # S = mc_spot(S0=S0, volatility=volatility0, T=T, r=r, N=N)

    # spot_name = './spot_folder/spot' + str(time.time()) + '.csv'
    # pd.DataFrame({'spot': S}).to_csv(spot_name)
    # print(spot_name)

    # S = pd.read_csv('./spot_folder/spot1626760012.523616.csv')['spot'].values

    # k_list = [24, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    k_list = [24, 12]
    error_list = []

    fig = make_subplots(rows=2, cols=3, specs=[[{}, {}, {}],
                                               [{"colspan": 3}, None, None]],
                        subplot_titles=("Spot histogram", "Error", "X", "Spot"),
                        row_heights=[0.7, 0.3])
    dt = []
    for k in k_list:
        S_ = S[::k]
        points_in_day = steps // k
        strikes_call = np.array([1.01 * S_[:-1:points_in_day]] * points_in_day).T.flatten()

        dt.append(T / len(S_))

        # volatility = np.std(np.log(S_[1:] / S_[:-1])) * np.sqrt(steps / k * 365)

        # print(S_[0], S_[-1])

        def count_delta():
            ds = 0.1

            price_call_S_plus_ds = np.array(
                [price_by_heston(strikes_call[i],
                                 call_expiration - (i % points_in_day) / points_in_day / 365,
                                 *params, S_[i] + ds, r, 'CALL')
                 for i in range(len(S_) - 1)])
            print('call +')
            price_call_S_minus_ds = np.array(
                [price_by_heston(strikes_call[i],
                                 call_expiration - (i % points_in_day) / points_in_day / 365,
                                 *params, S_[i] - ds, r, 'CALL')
                 for i in range(len(S_) - 1)])
            print('call -')
            price_put_S_plus_ds = np.array(
                [price_by_heston(put_strike,
                                 put_expiration - i / points_in_day / 365,
                                 *params, S_[i] + ds, r, 'PUT')
                 for i in range(len(S_) - 1)])
            print('put +')
            price_put_S_minus_ds = np.array(
                [price_by_heston(put_strike,
                                 put_expiration - i / points_in_day / 365,
                                 *params, S_[i] - ds, r, 'PUT')
                 for i in range(len(S_) - 1)])
            print('put -')
            delta_list = (price_call_S_plus_ds - price_call_S_minus_ds) / 2 / ds - \
                         (price_put_S_plus_ds - price_put_S_minus_ds) / 2 / ds

            return delta_list

        delta_list = count_delta()

        price_call_S = np.array(
            [price_by_heston(strikes_call[i],
                             call_expiration - (i % points_in_day) / points_in_day / 365,
                             *params, S_[i], r, 'CALL')
             for i in range(len(S_) - 1)])
        price_put_S = price_by_heston(put_strike,
                                      put_expiration,
                                      *params, S_[0], r, 'PUT')

        V0 = np.sum(price_call_S) - price_put_S

        intrinsic_call = np.array(
            [max(0, S_[i * points_in_day] - strikes_call[(i - 1) * points_in_day]) for i in range(1, 30)])
        intrinsic_put = max(0, put_strike - S_[-1])
        P = np.sum(intrinsic_call) - intrinsic_put

        R = np.array([delta_list[i - 1] * (S_[i] - S_[i - 1]) for i in range(1, len(S_))])

        X0 = (V0 - delta_list[0] * S0)
        X = [X0]
        for i in range(1, len(S_)):
            X1 = V0 + np.sum(R[:i]) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1) - delta_list[i] * S_[i]
            X.append(X1)

        print(np.sum(X) * (np.exp(r * (T / len(S_))) - 1))

        error1 = V0 + np.sum(R) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1) - P
        error2 = 100 * error1 / P
        error_list.append(error2)

        print()
        print('*' * 50)
        print(f'{steps // k} steps in a day')
        # print('Volatility', volatility)
        print('V0        ', V0)
        print('Sum(R_i)  ', np.sum(R))
        print('X0        ', X[0])
        print('Sum(X_i)  ', np.sum(X))
        print('X*exp  ', np.sum(X) * (np.exp(r * (T / len(S_))) - 1))
        print('P         ', P)
        pprint(f'Error      {error1:.6f}', 'y')
        pprint(f'Error      {error2:.6f}%', 'y')

        fig.add_trace(go.Histogram(x=np.log(S_ / S[0]), histnorm='probability', name=steps // k, opacity=0.5,
                                   xbins=dict(size=0.025)), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(0, len(S), k), y=X, name=steps // k, mode='lines+markers'), row=1, col=3)
        fig.add_trace(go.Scatter(x=np.arange(0, len(S), k), y=S_, name='spot' + str(steps // k), mode='lines'),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=dt, y=error_list, mode='lines+markers', name='error'), row=1, col=2)

    fig.update_xaxes(title_text="ln(S)", row=1, col=1)
    fig.update_xaxes(title_text="timestep", row=1, col=2)
    fig.update_yaxes(title_text="error", row=1, col=2)
    fig.update_yaxes(title_text="X", row=1, col=3)
    fig.update_yaxes(title_text="Spot", row=2, col=1)

    fig.show()  #


def get_heston_params_by_fitting_surface_from_paper():
    spot = 1
    r = 0.02

    expirations = np.array([30, 60, 90, 120, 150, 180, 252, 360]) / 365
    deltas = np.array([90, 75, 50, 25, 10]) / 100
    volatility_surface = np.array([[2.5096, 1.4359, 0.2808, 0.2540, 0.2369],
                                   [2.4351, 1.3216, 0.2847, 0.2606, 0.2417],
                                   [2.3823, 1.2955, 0.2878, 0.2660, 0.2489],
                                   [2.3383, 1.2677, 0.2904, 0.2699, 0.2548],
                                   [2.2996, 1.2407, 0.2925, 0.2745, 0.2598],
                                   [2.2619, 1.2166, 0.2943, 0.2777, 0.2641],
                                   [2.1767, 1.1671, 0.2975, 0.2837, 0.2722],
                                   [2.0618, 1.1136, 0.3007, 0.2897, 0.2803]])
    strikes = np.full(volatility_surface.shape, np.nan)
    price_list_BS = np.full(volatility_surface.shape, np.nan)
    for i, d in enumerate(deltas):
        for j, t in enumerate(expirations):
            strikes[j][i] = strike_from_delta(d, spot, t, volatility_surface[j][i], 'CALL', r)
            price_list_BS[j][i] = price_by_BS(S=spot, K=strikes[j][i], T=t, sigma=volatility_surface[j][i],
                                              option_type='CALL', r=r)

    bounds = np.array([[0.5, 5], [0.05, 0.95], [0.05, 0.95], [-0.9, -0.1], [0.05, 0.95]])
    # bounds = np.array([[-10, 10], [1e-3, 10], [1e-3, 10], [-1, 1], [1e-3, 10]])
    start_params = np.array([1.2, 0.20, 0.30, -0.60, 0.20])

    def objective_function_list(params):
        print(
            f'kappa {params[0]:.4f}, theta {params[1]:.4f}, sigma {params[2]:.4f}, rho {params[3]:.4f}, v0 {params[4]:.4f}',
            end=' --- ')
        price_list_H = np.full(volatility_surface.shape, np.nan)
        for i, d in enumerate(deltas):
            for j, t in enumerate(expirations):
                price_list_H[j][i] = price_by_heston(strike=strikes[j][i], T=t,
                                                     kappa=params[0], theta=params[1], sigma=params[2],
                                                     rho=params[3], v0=params[4], S0=spot, r=r, option_type='CALL')

        er = (np.array(price_list_BS) - np.array(price_list_H)).flatten()
        print(np.sqrt(np.sum(er ** 2)))
        return er

    res = least_squares(objective_function_list, start_params,
                        # method='dogbox',#
                        method='lm',
                        # bounds=bounds.T,
                        verbose=1)
    print(res)
    params = res.x
    price_list_H = np.full(volatility_surface.shape, np.nan)
    for i, d in enumerate(deltas):
        for j, t in enumerate(expirations):
            price_list_H[j][i] = price_by_heston(strike=strikes[j][i], T=t,
                                                 kappa=params[0], theta=params[1], sigma=params[2],
                                                 rho=params[3], v0=params[4], S0=spot, r=r, option_type='CALL')

    plot_price(price_list_BS, price_list_H, 'call', strikes, expirations)
    # fig = go.Figure()
    #
    # for n in range(len(expirations)):
    #     fig.add_trace(go.Scatter3d(
    #         x=strikes[n],
    #         y=[expirations[n] * 365] * len(strikes[n]),
    #         z=price_list_BS[n],
    #         name=f'BS {int(expirations[n] * 365)}',
    #         mode='markers', hovertext=deltas,
    #         marker=dict(size=3, color='green')))
    #
    #     fig.add_trace(go.Scatter3d(
    #         x=strikes[n],
    #         y=[expirations[n] * 365] * len(strikes[n]),
    #         z=price_list_H[n],
    #         name=f'BS {int(expirations[n] * 365)}',
    #         mode='markers', hovertext=deltas,
    #         marker=dict(size=3, color='red')))
    #
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(
    #             title='strike'
    #         ),
    #         yaxis=dict(
    #             title='expiration',
    #         ),
    #         zaxis=dict(
    #             title='price diff'
    #         ),
    #     )
    # )
    #
    # fig.show()


def get_heston_params_by_fitting_sabr_surface():
    spot = 31893.78
    r = 0.0
    bounds = np.array([[-10, 10], [1e-3, 10], [1e-3, 10], [-1, 1], [1e-3, 10]])
    start_params = np.array([1.2, 0.20, 0.30, -0.60, 0.20])
    volatility_surface = np.array([[2.63192848, 1.13835992, 0.74483752, 0.6500392, 0.74731848, 1.0717036, 2.0016311],
                                   [1.18089388, 0.76026322, 0.70171002, 0.69588187, 0.70649491, 0.76224645, 1.0370561],
                                   [1.2244828, 0.7883259, 0.72761139, 0.72156812, 0.73257291, 0.79038233, 1.07533571],
                                   [1.31166062, 0.84445126, 0.77941414, 0.77294061, 0.7847289, 0.8466541, 1.15189491],
                                   [0.97434825, 0.84043515, 0.81096446, 0.79764685, 0.78527263, 0.76334716,
                                    0.71512297]])

    # np.array([[0.69711486, 0.34569527, 0.27321177, 0.26758712, 0.29026554, 0.36583683, 0.61209884],
    #                            [0.71361237, 0.35387532, 0.27967649, 0.27391866, 0.2971336, 0.37449278, 0.6265804],
    #                            [0.73423426, 0.36410038, 0.28775738, 0.28183308, 0.30571866, 0.38531272, 0.64468234],
    #                            [0.77547804, 0.38455049, 0.30391916, 0.29766192, 0.3228888, 0.40695259, 0.68088621],
    #                            [0.81672181, 0.4050006, 0.32008094, 0.31349076, 0.34005893, 0.42859246, 0.71709009]])

    price_list_BS = np.full(volatility_surface.shape, np.nan)
    price_list_BS_C = np.full(volatility_surface.shape, np.nan)
    price_list_BS_P = np.full(volatility_surface.shape, np.nan)
    print(price_list_BS.shape)
    strikes_list = np.array([22000.0, 29000.0, 31000.0, 32000.0, 33000.0, 35000.0, 42000.0])
    times_list = np.array(
        [0.0027397260273972603, 0.0136986301369863, 0.0273972602739726, 0.0547945205479452, 0.0821917808219178])

    for i, strike in enumerate(strikes_list):
        for j, t in enumerate(times_list):
            if strike > spot:
                option_type = 'PUT'
            else:
                option_type = 'CALL'
            price_list_BS[j][i] = price_by_BS(S=spot, K=strike, T=t, sigma=volatility_surface[j][i],
                                              option_type=option_type, r=r)
            price_list_BS_C[j][i] = price_by_BS(S=spot, K=strike, T=t, sigma=volatility_surface[j][i],
                                                option_type='CALL', r=r)
            price_list_BS_P[j][i] = price_by_BS(S=spot, K=strike, T=t, sigma=volatility_surface[j][i],
                                                option_type='PUT', r=r)

    print(price_list_BS)

    def objective_function_list(params):
        print(
            f'kappa {params[0]:.4f}, theta {params[1]:.4f}, sigma {params[2]:.4f}, rho {params[3]:.4f}, v0 {params[4]:.4f}',
            end=' --- ')
        price_list_H = np.full(volatility_surface.shape, np.nan)
        for i, strike in enumerate(strikes_list):
            for j, t in enumerate(times_list):
                if strike > spot:
                    option_type = 'PUT'
                else:
                    option_type = 'CALL'
                price_list_H[j][i] = price_by_heston(strike=strike, T=t,
                                                     kappa=params[0], theta=params[1], sigma=params[2],
                                                     rho=params[3], v0=params[4], S0=spot, r=r, option_type=option_type)
        er = (np.array(price_list_BS) - np.array(price_list_H)).flatten()
        print(np.sqrt(np.sum(er ** 2)))
        return er

    res = least_squares(objective_function_list, start_params,
                        method='dogbox',
                        bounds=bounds.T,
                        # method='lm',
                        verbose=1)  #
    print(res)

    params = res.x
    # params = [0.34823362, 10., 10., 0.13048711, 0.08991543]

    price_list_H_C = np.full(volatility_surface.shape, np.nan)
    price_list_H_P = np.full(volatility_surface.shape, np.nan)
    for i, strike in enumerate(strikes_list):
        for j, t in enumerate(times_list):
            if strike > spot:
                option_type = 'PUT'
            else:
                option_type = 'CALL'
            price_list_H_C[j][i] = price_by_heston(strike=strike, T=t,
                                                   kappa=params[0], theta=params[1], sigma=params[2],
                                                   rho=params[3], v0=params[4], S0=spot, r=r, option_type='CALL')
            price_list_H_P[j][i] = price_by_heston(strike=strike, T=t,
                                                   kappa=params[0], theta=params[1], sigma=params[2],
                                                   rho=params[3], v0=params[4], S0=spot, r=r, option_type='PUT')

    # print(price_list_H_C)
    plot_price(price_list_BS_C, price_list_H_C, 'call', strikes_list, times_list[1:])
    plot_price(price_list_BS_P, price_list_H_P, 'put', strikes_list, times_list[1:])


def get_bs_prices_for_1d_30d_options(spot, call_strike, put_strike, call_vol, put_vol, r,
                                     expiration_1d=1 / 365, expiration_30d=30 / 365,
                                     is_print=True):
    call_1d = price_by_BS(S=spot, K=call_strike, T=expiration_1d, sigma=call_vol, option_type='CALL', r=r)
    put_1d = price_by_BS(S=spot, K=put_strike, T=expiration_1d, sigma=put_vol, option_type='PUT', r=r)

    call_30d = price_by_BS(S=spot, K=call_strike, T=expiration_30d, sigma=call_vol, option_type='CALL', r=r)
    put_30d = price_by_BS(S=spot, K=put_strike, T=expiration_30d, sigma=put_vol, option_type='PUT', r=r)

    if is_print:
        print('*' * 100, '\nBS PRICES')
        print(f'call 1d  {call_1d:.4f}\ncall 30d {call_30d:.4f}\nput 1d   {put_1d:.4f}\nput 30d  {put_30d:.4f}')

    return call_1d, put_1d, call_30d, put_30d


def get_heston_prices_for_1d_30d_options(spot, call_strike, put_strike, r, params,
                                         expiration_1d=1 / 365, expiration_30d=30 / 365,
                                         is_print=True):
    call_1d = price_by_heston(strike=call_strike, T=expiration_1d,
                              kappa=params[0], theta=params[1], sigma=params[2],
                              rho=params[3], v0=params[4], S0=spot, r=r,
                              option_type='CALL')
    call_30d = price_by_heston(strike=call_strike, T=expiration_30d,
                               kappa=params[0], theta=params[1], sigma=params[2],
                               rho=params[3], v0=params[4], S0=spot, r=r,
                               option_type='CALL')

    put_1d = price_by_heston(strike=put_strike, T=expiration_1d,
                             kappa=params[0], theta=params[1], sigma=params[2],
                             rho=params[3], v0=params[4], S0=spot, r=r,
                             option_type='PUT')
    put_30d = price_by_heston(strike=put_strike, T=expiration_30d,
                              kappa=params[0], theta=params[1], sigma=params[2],
                              rho=params[3], v0=params[4], S0=spot, r=r,
                              option_type='PUT')

    if is_print:
        print('*' * 100, '\nHESTON PRICES')
        print(f'call 1d  {call_1d:.4f}\ncall 30d {call_30d:.4f}\nput 1d   {put_1d:.4f}\nput 30d  {put_30d:.4f}')

    return call_1d, put_1d, call_1d, put_30d


def heston_mc(S, call_strike, put_strike, r,
              expiration_1d=1 / 365, expiration_30d=30 / 365,
              is_print=True):
    # mc_1d = price_by_heston_mc(call_strike=call_strike, put_strike=put_strike, T=expiration_1d,
    #                            kappa=params[0], theta=params[1], sigma=params[2], rho=params[3], v0=params[4],
    #                            S0=spot, r=r, M=M)
    # mc_30d = price_by_heston_mc(call_strike=call_strike, put_strike=put_strike, T=expiration_30d,
    #                             kappa=params[0], theta=params[1], sigma=params[2], rho=params[3], v0=params[4],
    #                             S0=spot, r=r, M=M)

    mc_1d = price_by_heston_mc(call_strike=call_strike, put_strike=put_strike, days=1, S=S, r=r)
    mc_30d = price_by_heston_mc(call_strike=call_strike, put_strike=put_strike, days=30, S=S, r=r)

    call_1d = mc_1d[0]
    call_30d = mc_30d[0]

    put_1d = mc_1d[1]
    put_30d = mc_30d[1]

    if is_print:
        print('*' * 100, '\nMC HESTON PRICES')
        print(f'call 1d  {call_1d:.4f}\ncall 30d {call_30d:.4f}\nput 1d   {put_1d:.4f}\nput 30d  {put_30d:.4f}')

    return call_1d, put_1d, call_1d, put_30d


def simple_hedging_BS(S, r, params, days_count, call_expiration=30 / 365, put_expiration=30 / 365, plot=False):
    call_strike = 29000.0
    volatility0 = np.sqrt(0.08991543)  # 0.26758712

    S0 = 31893.78

    steps = 24 * 12
    days = days_count
    T = days / 365
    N = days * steps
    r = 0.0

    # np.random.seed(42)

    S = spot_bs_mc(days=days, S0=S0, volatility=volatility0, r=r)
    pprint(f'len(S) : {len(S)}', 'c')

    spot_name = './spot_folder/spot' + str(time.time()) + '.csv'
    pd.DataFrame({'spot': S}).to_csv(spot_name)
    print(spot_name)

    # S = pd.read_csv('./spot_folder/spot1626760012.523616.csv')['spot'].values

    # k_list = [24, 12, 6, 4, 3, 2, 1]
    # k_list = [24, 12, 8, 4, 2, 1]

    k_list = [12]
    error_list = []
    if plot:
        fig = make_subplots(rows=3, cols=3, specs=[[{}, {}, {}],
                                                   [{"colspan": 3}, None, None],
                                                   [{"colspan": 3}, None, None]],
                            subplot_titles=("Spot returns histogram", "Error", "X", "Spot", 'Delta'),
                            row_heights=[0.5, 0.25, 0.25])
    dt_list = []
    for k in k_list:
        if (len(S) - 1) % k == 0:
            print()
            print('*' * 50)

            S_ = S[::k]

            pprint(f'len(S_) : {len(S_)}', 'c')
            pprint(f'{steps // k} steps in a day, k = {k}', 'c')

            dt = T / (len(S_) - 1)
            dt_list.append(dt)

            volatility = np.std(np.log(S_[1:] / S_[:-1])) * np.sqrt(steps / k * 365)

            ds = 0.1

            price_call_S_plus_ds = np.array(
                [price_by_BS(S_[i] + ds, call_strike, T - i * dt_list[-1], volatility, option_type='CALL', r=r)
                 for i in range(len(S_) - 1)])

            price_call_S_minus_ds = np.array(
                [price_by_BS(S_[i] - ds, call_strike, T - i * dt_list[-1], volatility, option_type='CALL', r=r)
                 for i in range(len(S_) - 1)])
            delta_list = (price_call_S_plus_ds - price_call_S_minus_ds) / (2 * ds)
            if delta_list[-1] >= 0.9:
                delta_list = np.append(delta_list, 1)
            else:
                delta_list = np.append(delta_list, 0)
            delta_list[delta_list > 1] = 1

            price_call_S = np.array(
                [price_by_BS(S_[i], call_strike, T - i * dt_list[-1], volatility, option_type='CALL', r=r)
                 for i in range(len(S_) - 1)])
            pprint('len(price), len(delta) ' + str((len(price_call_S), len(delta_list))), 'm')

            V0 = price_call_S[0]

            R = np.array([delta_list[i - 1] * (S_[i] - S_[i - 1]) for i in range(1, len(S_))])
            pprint('len(R) ' + str((len(R))), 'm')

            P = max(0, S_[-1] - call_strike)

            X0 = (V0 - delta_list[0] * S0)
            X = []
            for i in range(1, len(S_)):
                if i == 1:
                    X1 = V0 + R[0] + X0 * (np.exp(r * (T / len(S_))) - 1) - delta_list[1] * S_[1]
                else:
                    X1 = V0 + np.sum(R[:i]) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1) - delta_list[i] * S_[i]
                X.append(X1)
            pprint('len(X) ' + str((len(X))), 'm')

            print(np.sum(X) * (np.exp(r * (T / len(S_))) - 1))
            error1 = V0 + np.sum(R) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1) - P
            error2 = 100 * error1 / call_strike
            error_list.append(error2)

            print('Volatility', volatility)
            print('V0        ', V0)
            print('Sum(R_i)  ', np.sum(R))
            print('X0        ', X[0])
            print('Sum(X_i)  ', np.sum(X))
            print('X*exp  ', np.sum(X) * (np.exp(r * (T / len(S_))) - 1))
            print('P         ', P)
            pprint(f'Error      {error1:.6f}', 'y')
            pprint(f'Error      {error2:.6f}%', 'y')
            if plot:

                fig.add_trace(go.Histogram(x=np.log(S_[1:] / S_[:-1]), histnorm='probability', name=steps // k, opacity=0.5,
                                           xbins=dict(size=0.01)), row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(0, len(S), k), y=X, name=steps // k, mode='lines+markers'),
                              row=1, col=3)
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(S), k), y=S_, name='spot' + str(steps // k), mode='lines+markers'),
                    row=2, col=1)
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(S), k), y=delta_list, name='delta ' + str(steps // k),
                               mode='lines+markers'),
                    row=3, col=1)
    if plot:

        fig.add_trace(go.Scatter(x=dt_list, y=np.abs(error_list), mode='lines+markers', name='error'), row=1, col=2)
        fig.update_xaxes(title_text="ln(returns)", row=1, col=1)
        fig.update_xaxes(title_text="timestep", row=1, col=2)
        fig.update_yaxes(title_text="error", row=1, col=2)
        fig.update_yaxes(title_text="X", row=1, col=3)
        fig.update_yaxes(title_text="Spot", row=2, col=1)
        fig.update_layout(title_text=f"BS, spot {S0}, strike {call_strike}, r {r}, volatility {volatility0}, days {days}")

        fig.show()

    return np.abs(100 * error1 / call_strike)


def simple_hedging_HESTON(S, r, params, days_count, call_expiration=30 / 365, put_expiration=30 / 365, it=0, fig=None, plot=False):
    call_strike = 29000.0

    S0 = 31893.78

    pprint(f'len(S) : {len(S)}', 'c')

    steps = 24 * 12
    days = days_count
    print('DAYS', days)
    T = days / 365
    N = days * steps
    r = 0.0

    # k_list = [24, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    # k_list = [24, 12, 8, 4, 2, 1]
    k_list = [24 * 12, 24 * 6, 24, 12]
    # k_list = [12]
    error_list = []

    if plot and fig is None:
        fig = make_subplots(rows=3, cols=3, specs=[[{}, {}, {}],
                                                   [{"colspan": 3}, None, None],
                                                   [{"colspan": 3}, None, None]],
                            subplot_titles=("Spot returns histogram", "Error", "X", "Spot", 'Delta'),
                            row_heights=[0.5, 0.25, 0.25])
    dt_list = []
    for k in k_list:
        if (len(S) - 1) % k == 0:
            print()
            print('*' * 50)

            S_ = S[::k]

            pprint(f'len(S_) : {len(S_)}', 'c')
            pprint(f'{steps // k} steps in a day, k = {k}', 'c')

            dt = T / (len(S_) - 1)
            dt_list.append(dt)

            print('S', S_[0], S_[-1])
            # print('T - i*dt_list',
            #       [T - i * dt for i in range(len(S_) - 1)][0],
            #       [T - i * dt for i in range(len(S_) - 1)][-2],
            #       [T - i * dt for i in range(len(S_) - 1)][-1])

            ds = 0.05

            price_call_S_plus_ds = np.array(
                [price_by_heston(call_strike, T - i * dt, *params, S_[i] + ds, r, 'CALL')
                 for i in range(len(S_) - 1)])
            print('call +')

            price_call_S_minus_ds = np.array(
                [price_by_heston(call_strike, T - i * dt, *params, S_[i] - ds, r, 'CALL')
                 for i in range(len(S_) - 1)])

            print('call -')

            delta_list = (price_call_S_plus_ds - price_call_S_minus_ds) / (2 * ds)
            if delta_list[-1] >= 0.9:
                delta_list = np.append(delta_list, 1)
            else:
                delta_list = np.append(delta_list, 0)
            delta_list[delta_list > 1] = 1
            # delta_list[delta_list < 0] = 0

            price_call_S = np.array(
                [price_by_heston(call_strike, T - i * dt, *params, S_[i], r, 'CALL')
                 for i in range(len(S_) - 1)])
            print('call')

            V0 = price_call_S[0]

            R = np.array([delta_list[i - 1] * (S_[i] - S_[i - 1]) for i in range(1, len(S_))])

            P = max(0, S_[-1] - call_strike)

            X0 = V0 - delta_list[0] * S0
            X = []

            for i in range(1, len(S_)):
                if i == 1:
                    X1 = V0 + R[0] + X0 * (np.exp(r * (T / len(S_))) - 1) - delta_list[1] * S_[1]
                else:
                    X1 = V0 + np.sum(R[:i]) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1) - delta_list[i] * S_[i]
                X.append(X1)

            print(np.sum(X) * (np.exp(r * (T / len(S_))) - 1))
            V_N = V0 + np.sum(R) + np.sum(X) * (np.exp(r * (T / len(S_))) - 1)
            error1 = V_N - P
            error2 = np.abs(100 * error1) / call_strike
            error_list.append(100 * error1 / call_strike)

            print('V0        ', np.round(V0, 3))
            print('Sum(R_i)  ', np.round(np.sum(R), 3))
            print('X0        ', np.round(X[0], 3))
            print('Sum(X_i)  ', np.round(np.sum(X), 3))
            print('X*exp  ', np.round(np.sum(X) * (np.exp(r * (T / len(S_))) - 1), 3)),
            print('P         ', np.round(P, 3))
            pprint(f'Error      {error1:.3f}$', 'y')
            if np.abs(error2) > 1:
                c = 'r'
            else:
                c = 'g'
            pprint(f'Error      {error2:.3f}%', c)
            if plot and fig is None:
                hist_data = np.log(S_[1:] / S_[:-1])
                fig.add_trace(
                    go.Histogram(x=hist_data, histnorm='probability', name=steps // k, opacity=0.5,
                                 xbins=dict(size=0.01)),
                    row=1, col=1)
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(S), k), y=X, name=steps // k, mode='lines+markers'),
                    row=1, col=3)
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(S), k), y=S_, name='spot' + str(steps // k), mode='lines+markers'),
                    row=2, col=1)
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(S), k), y=delta_list, name='delta ' + str(steps // k),
                               mode='lines+markers'),
                    row=3, col=1)
    if plot and fig is None:
        fig.add_trace(go.Scatter(x=dt_list, y=np.abs(error_list), mode='lines+markers', name='error'), row=1, col=2)

        fig.update_xaxes(title_text="ln(S)", row=1, col=1)
        fig.update_xaxes(title_text="timestep", row=1, col=2)
        fig.update_yaxes(title_text="error", row=1, col=2)
        fig.update_yaxes(title_text="X", row=1, col=3)
        fig.update_yaxes(title_text="Spot", row=2, col=1)
        fig.update_layout(title_text=f"Heston, spot {S0}, strike {call_strike}, r {r}, params {params}, days {days}")

        fig.show()

    if fig is not None:
        fig.add_trace(
            go.Scatter(x=dt_list, y=np.abs(error_list), mode='lines+markers', name=f'error, it = {it}', opacity=0.7))

    return np.abs(100 * error1 / call_strike), np.abs(error_list), dt_list


def bs_heston_prices_for_different_maturity():
    spot = 31893.78
    call_strike = 32000.0
    r = 0.0

    bs_vols = [0.6500392, 0.69588187, 0.72156812, 0.77294061, 0.79764685]
    params = [-8.98698206e+00, 1.00000000e-03, 4.81155610e+00, -1.63466446e-01, 5.66211348e-01]
    bs_prices = []
    h_prices = []
    days = np.array(
        [0.0027397260273972603, 0.0136986301369863, 0.0273972602739726, 0.0547945205479452, 0.0821917808219178])
    for n, t in enumerate(days):
        T = t
        h_prices.append(price_by_heston(call_strike, T, *params, spot, r, 'CALL'))
        bs_prices.append(price_by_BS(spot, call_strike, T, bs_vols[n], option_type='CALL', r=r))

    plt.plot(days, h_prices, label='Heston')
    plt.plot(days, bs_prices, label='BS')
    plt.legend()
    plt.show()
    print(h_prices)
    print(bs_prices)


def count_mean_error_by_heston_hedging():
    # params = [-8.98698206e+00, 1.00000000e-03, 4.81155610e+00, -1.63466446e-01, 5.66211348e-01]
    params = [0, 1.00000000e-03, 1.81155610e+00, -1.63466446e-01, 5.66211348e-01]
    fig = go.Figure()
    error_matrix = []
    days_count = 30
    spot = 31893.78
    r = 0.0
    N = 1000
    for it in range(N):
        pprint(f'ITERATION № {it}, heston iterations 10000', color='m')
        S = spot_heston_mc(days=days_count,
                           kappa=params[0], theta=params[1], sigma=params[2], rho=params[3], v0=params[4],
                           S0=spot, r=r, M=10000)
        _, error_list, x = simple_hedging_HESTON(S, r, params, days_count, it=it, fig=fig, plot=False)
        error_matrix.append(error_list)

    fig.add_trace(
        go.Scatter(x=x, y=np.mean(error_matrix, axis=0), mode='lines+markers', name=f'Mean error',
                   marker=dict(color='black')))
    fig.update_layout(title=f'Days count {days_count}, {params}, N = {N}')

    fig.show()
    print()
    print('*' * 20)
    print('*' * 20)
    pprint(f'Days count {days_count}', 'm')
    pprint(f'Mean error {np.mean(error_matrix, axis=0)}', 'm')
    print('*' * 20)
    print('*' * 20)


def count_heston_hedging_by_different_sigma():
    params = [-8.98698206e+00, 1.00000000e-03, 4.81155610e+00, -1.63466446e-01, 5.66211348e-01]
    days_count = 30
    spot = 31893.78
    r = 0.0
    sigma_list = [1e-6, 1e-3, 1e-1, 0.5, 1, 2.5, params[2]]
    h_error = []
    bs_error = []
    for sigma in sigma_list:
        params[2] = sigma
        S = spot_heston_mc(days=days_count,
                           kappa=params[0], theta=params[1], sigma=params[2], rho=params[3], v0=params[4],
                           S0=spot, r=r, M=5000)
        h_error.append(simple_hedging_HESTON(S, r, params, days_count))
        bs_error.append(simple_hedging_BS(S, r, params, days_count, plot=False))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sigma_list, y=bs_error, name='error by BS',
                   mode='lines+markers'),
    )
    fig.add_trace(
        go.Scatter(x=sigma_list, y=h_error, name='error by Heston',
                   mode='lines+markers'),
    )
    fig.update_xaxes(title_text="sigma")
    fig.update_yaxes(title_text="Error")
    fig.show()


def hedging_heston_or_bs_prices_by_heston_spot(hedging_method='Heston'):
    np.set_printoptions(suppress=True)

    spot = 31893.78
    vol = 0.26758712
    r = 0.0

    days_count = 30

    # params = [0.34823362, 10., 10., 0.13048711, 0.08991543]
    # params = [0.34823362, 1., 1, 0.13048711, 0.08991543]
    params = [-8.98698206e+00, 1.00000000e-03, 4.81155610e+00, -1.63466446e-01, 5.66211348e-01]

    print(f'{"*" * 100}\nPARAMS: kappa {params[0]:.4f}, theta {params[1]:.4f}, sigma {params[2]:.4f},',
          f'rho {params[3]:.4f}, v0 {params[4]:.4f}')
    # np.random.seed(42)

    print('*' * 100, '\nSPOT SAVING')
    S = spot_heston_mc(days=days_count,
                       kappa=params[0], theta=params[1], sigma=params[2], rho=params[3], v0=params[4],
                       S0=spot, r=r, M=10000)
    np.save('MC_spot.npy', S)
    S = np.load('MC_spot.npy')

    if hedging_method == 'Heston':
        simple_hedging_HESTON(S, r, params, days_count, fig=None, plot=True)
    elif hedging_method == 'BS':
        simple_hedging_BS(S, r, params, days_count, plot=True)
    else:
        hedging_HESTON(S, r, params)


def awful_thing():
    call_strike = 32000.0  # 1.05 * spot
    put_strike = 22000.0  # 0.9 * spot

    call_vol = 0.26758712
    put_vol = 0.81672181

    expiration_call = 1 / 365
    expiration_put = 30 / 365

    # bs_call_short, bs_put_short, bs_call_long, bs_put_long = \
    #     bs(spot, call_strike, put_strike, call_vol, put_vol, r)

    # h_call_short, h_put_short, h_call_long, h_put_long = \
    #     heston(spot, call_strike, put_strike, r, params)

    # h_mc_call_short, h_mc_put_short, h_mc_call_long, h_mc_put_long = \
    #     heston_mc(S, call_strike, put_strike, r)
    # res = hedging_product_heston(call_strike, put_strike, S[index], r, params)

    # res = hedging_product_heston2(put_strike, S[index], r, params)

    # res = hedging_product_heston_df(S[index], r, params)

    # np.save('save1.npy', res)
    #
    # res = np.load('save1.npy', allow_pickle=True)
    #
    # print('Hedge result (sum)', res[0])
    # print('Intrinsic put', res[3])
    #

    # a = np.arange(720)
    #
    # intr = [np.nan if i not in list(a[24::24]) + list([a[-1]]) else res[2][i // 24 - 1] for i in list(a[1::])]
    # intr[-1] = res[2][-1]
    #
    # print(len(intr), len(res[1]))
    # pd.DataFrame({'Hedge result': res[1], 'Intrinsic call': intr}).to_csv('hedge_heston.csv')
    # plt.plot(res[4], label='delta call')
    # plt.plot(res[5], label='delta put')
    # plt.legend()
    # plt.show()
    # print('*' * 100, '\nPRODUCT')
    # res = product_hestonMC(call_strike=call_strike, put_strike=put_strike, days=days_count, S=S, r=r)
    # res = hedging_product_hestonMC(call_strike=call_strike, put_strike=put_strike, days=days_count, S=S, r=r)

    # print()
    # print(*res)

    # heston_prices_for_mc_spot = []
    # for s in res[-1][24::24]:
    #     heston_prices_for_mc_spot.append(price_by_heston(strike=call_strike, T=1 / 365,
    #                                                      kappa=params[0], theta=params[1], sigma=params[2],
    #                                                      rho=params[3], v0=params[4], S0=s, r=r,
    #                                                      option_type='CALL'))

    # plt.plot(heston_prices_for_mc_spot)
    # plt.plot(res[0])
    # plt.show()

    # delta_of_product = delta_by_finite_difference(days=days_count, call_strike=call_strike, put_strike=put_strike,
    #                                               spot=spot, r=r, params=params, price_by_spot=res)
    # print('Delta of product', delta_of_product)


if __name__ == '__main__':
    # params = [0.34823362, 10., 10., 0.13048711, 0.08991543]
    # S = 32893.78

    # функции для подбора параметров
    # get_heston_params_by_fitting_sabr_surface()
    # get_heston_params_by_fitting_surface_from_paper()
    # get_heston_params_from_2_1d_options_and_count_prices_for_different_strikes_and_expirations()

    # просто считаем цены двумя методами
    # bs_heston_prices_for_different_maturity()

    # функции для хэджирования с разными сигмами/спотами
    count_mean_error_by_heston_hedging()
    # count_heston_hedging_by_different_sigma()

    # функция, вызывающая хэджирование разными методами
    # hedging_heston_or_bs_prices_by_heston_spot(hedging_method='Heston')

    # что-то с чем-то
    # awful_thing()
