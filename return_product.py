import numpy as np
import pandas as pd
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px

import argparse

from lib.option_formulas import price_by_BS


def get_heston_params():
    df = pd.read_csv('heston_params.csv')
    return df.kappa.values[0], df.theta.values[0], df.sigma.values[0], df.rho.values[0], df.v0.values[0]


def get_real_params_from_historical_spot():
    df = pd.read_csv(
        'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\old_complect\\BTCUSDT.csv',
        usecols=[2], header=None)[::60]

    S = np.array([df[df.columns[0]].values[-6 * 30 * 24 - 1:]])

    print(S.shape)

    df = df[::24]

    vol = (np.log(df[df.columns[0]] / df[df.columns[0]].shift(1)).rolling(30).std() * np.sqrt(365))
    v = (np.log(df[df.columns[0]] / df[df.columns[0]].shift(1)).std() * np.sqrt(365))
    sigma = np.log(vol / vol.shift(1)).std() * np.sqrt(365)
    rho = np.corrcoef(df[df.columns[0]][30:], vol[30:])[0][1]
    theta = np.mean(vol[30:])
    v0 = v ** 2

    print(theta, sigma, rho, v0)

    plt.plot(vol)
    plt.show()


def spot_bs_mc(S0, volatility, r=0, days=6 * 30, M=1000, steps=24):
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


def spot_heston_mc(days, kappa, theta, sigma, rho, v0, S0, r, M=500, steps=24 * 12):
    delta_t = 1 / 365 / steps

    N = days * steps + 1

    S = np.full((M, N), S0, dtype=float)
    V = np.full((M, N), v0, dtype=float)

    # S = np.full(N, S0, dtype=float)
    print(f'SPOT CALCULATING --- {M} iterations')
    for m in range(M):
        check_not_finish = True
        count_nan = 0
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

                S[m][n] = St
                V[m][n] = vt
            if not np.isnan(np.sum(S[m])):
                check_not_finish = False
            else:
                count_nan += 1
        if count_nan > 0:
            print(f'Count of bad attempts for one iteration {count_nan}, iteration {m}')
    return S, V


def get_product_matrices(payment_percent, call_percent, S, V, r=0, days=6 * 30, steps=24, vol_method='Heston_vol'):
    M, N = S.shape

    strikes_matrix = np.full((M, days), np.nan)
    intrinsic_matrix = np.full((M, days), np.nan)
    if vol_method != 'Heston_vol':
        calculated_vol_matrix = np.full((M, days), np.nan)
        bs_price_matrix_MC_vol = np.full((M, days), np.nan)
    else:
        heston_vol_matrix = V[:, ::steps]
        bs_price_matrix_Heston_vol = np.full((M, days), np.nan)
    client_money_matrix = np.full((M, days), np.nan)
    tokens_matrix = np.full((M, days), np.nan)

    for m in range(M):
        strikes_matrix[m] = np.array(
            [call_percent * s for s in S[m][:-steps:steps]])
        intrinsic_matrix[m] = np.array(
            [np.exp(-1 / 365 * r) * max(0, s - k) for s, k in zip(S[m][steps::steps], strikes_matrix[m])])

        if vol_method != 'Heston_vol':
            calculated_vol_matrix[m] = np.array(
                [np.std(np.log(S[m][1:(n + 1) * steps] / S[m][:(n + 1) * steps - 1])) * np.sqrt(steps * 365)
                 for n in range(days)])

            bs_price_matrix_MC_vol[m] = np.array(
                [price_by_BS(S[m][n * steps], strikes_matrix[m][n], 1 / 365, calculated_vol_matrix[m][n], 'CALL')
                 for n in range(days)]
            )
        else:
            bs_price_matrix_Heston_vol[m] = np.array(
                [price_by_BS(S[m][n * steps], strikes_matrix[m][n], 1 / 365, heston_vol_matrix[m][n], 'CALL')
                 for n in range(days)]
            )

        # heston_price_matrix[m] = np.array(
        #     [price_by_heston(
        #         strikes_matrix[m][n],
        #         1 / 365,
        #         params[0], params[1], params[2], params[3], heston_vol_matrix[m][n] ** 2,
        #         S[m][n * steps],
        #         r=0,
        #         option_type='CALL') for n in range(days)]
        # )

        client_money_matrix[m] = np.array(
            [payment_percent * s for s in S[m][:-1:steps]])

    for m in range(M):
        one_iteration_tokens = np.ones(days)

        t = 1
        for n in range(1, days):
            one_iteration_tokens[n] = t * strikes_matrix[m][n] / S[m][(n + 1) * steps] \
                if intrinsic_matrix[m][n] > 0 else t
            t = one_iteration_tokens[n]
            # one_iteration_tokens[n] += t * option_price_list[n] / S[m][n * steps]

        tokens_matrix[m] = one_iteration_tokens
    if vol_method != 'Heston_vol':
        return S, strikes_matrix, calculated_vol_matrix, intrinsic_matrix, bs_price_matrix_MC_vol, client_money_matrix, tokens_matrix, vol_method
    else:
        return S, strikes_matrix, heston_vol_matrix, intrinsic_matrix, bs_price_matrix_Heston_vol, client_money_matrix, tokens_matrix, vol_method


def plot_and_write_product_prices(S, strikes_matrix, vol_matrix, intrinsic_matrix,
                                  bs_price_matrix,
                                  client_money_matrix, tokens_matrix, vol_method='Heston_vol',
                                  plot_prices=False, plot_vol=False):
    M, days = strikes_matrix.shape

    x = np.arange(days)

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.05,
        subplot_titles=[' ', ' '],
    )

    final_price_matrix = bs_price_matrix * tokens_matrix
    final_client_money_matrix = client_money_matrix * tokens_matrix

    def count_product_price_by_nan(price_matrix):
        price_matrix_ = price_matrix.copy()
        client_money_matrix_ = client_money_matrix.copy()
        tokens_matrix_ = tokens_matrix.copy()
        for m in range(M):
            indices = np.argwhere(np.nancumsum(price_matrix_[m]) < np.nancumsum(client_money_matrix_[m]))

            if len(indices) > 0:
                i = indices[0][0]
                price_matrix_[m][i:] = np.nan
                client_money_matrix_[m][i:] = np.nan
                tokens_matrix_[m][i:] = np.nan

        return np.nansum(
            np.nanmean(price_matrix_ * tokens_matrix_, axis=0) - np.nanmean(client_money_matrix_ * tokens_matrix_,
                                                                            axis=0)), \
               price_matrix_ * tokens_matrix_, client_money_matrix_ * tokens_matrix_

    price_nan, final_price_matrix_, final_client_money_matrix_ = count_product_price_by_nan(
        price_matrix=bs_price_matrix)

    # price_nan_intr, final_price_matrix_intr, final_client_money_matrix_intr = count_product_price_by_nan(
    #     price_matrix=intrinsic_matrix)

    def results_to_csv(print_df=False):
        df = pd.DataFrame()
        df['Options_prices'] = [np.sum(np.mean(final_price_matrix, axis=0))]
        df['Clients_money'] = [np.sum(np.mean(final_client_money_matrix, axis=0))]
        df['Product_price'] = [
            np.sum(np.mean(final_price_matrix, axis=0)) - np.sum(np.mean(final_client_money_matrix, axis=0))]

        df['Options_prices_before_stop'] = [np.nansum(np.nanmean(final_price_matrix_, axis=0))]
        df['Clients_money_before_stop'] = [np.nansum(np.nanmean(final_client_money_matrix_, axis=0))]
        df['Product_price_before_stop'] = [price_nan]
        df.to_csv('return_product_results.csv', index=False)
        if print_df:
            print(df)

    results_to_csv(False)

    title = f'Product price by {vol_method} ' + str(np.round(
        np.sum(np.mean(final_price_matrix, axis=0) - np.mean(final_client_money_matrix, axis=0)), 2)) \
            + ', Stop price ' + str(np.round(price_nan, 2))

    colors = px.colors.qualitative.Plotly * 1000

    if plot_vol:
        def plot_option_prices_and_vol(m):
            fig.add_trace(
                go.Scatter(x=x, y=vol_matrix[m],
                           name=f'{m} {vol_method}',
                           mode='lines', marker=dict(color=colors[m])),
                row=1, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=bs_price_matrix[m] * tokens_matrix[m],
                           name=f'{m} Price by {vol_method}',
                           mode='lines', marker=dict(color=colors[m])),
                row=2, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=client_money_matrix[m] * tokens_matrix[m],
                           name=f'{m} Client money',
                           mode='markers+lines', marker=dict(color=colors[m])),
                row=2, col=1)

        for m in range(M):
            plot_option_prices_and_vol(m)

        fig.update_layout(title_text=title)
        fig.show()

    def plot_product_prices():
        fig = go.Figure()
        for m in range(M):
            fig.add_trace(
                go.Scatter(x=x, y=np.nancumsum(final_price_matrix[m] - final_client_money_matrix[m]),
                           name=f'{m} Difference',
                           mode='lines', line=dict(color=colors[m]), opacity=0.4))
            # fig.add_trace(
            #     go.Scatter(x=x, y=np.nancumsum((intrinsic_matrix * tokens_matrix)[m] - final_client_money_matrix[m]),
            #                name=f'{m} Difference by intr',
            #                mode='lines', line=dict(color='red'), opacity=0.4))

        fig.add_trace(
            go.Scatter(x=x, y=np.nancumsum(np.mean(final_price_matrix - final_client_money_matrix, axis=0)),
                       name=f'Mean difference',
                       mode='lines', line=dict(color='black')))
        # fig.add_trace(
        #     go.Scatter(x=x,
        #                y=np.nancumsum(np.mean(intrinsic_matrix * tokens_matrix - final_client_money_matrix, axis=0)),
        #                name=f'Mean difference by intrinsic price',
        #                mode='lines', line=dict(color='red')))

        fig.update_layout(title_text=title)
        fig.show()

    if plot_prices:
        plot_product_prices()


def get_product_price(payment_percent, call_percent, spot, r, vol_method='Heston_vol',
                      check_if_spot_is_ready=False, plot_prices=False, plot_vol=False):
    np.set_printoptions(suppress=True)

    params = get_heston_params()

    spot_name = 'Heston_spot.npy'
    vol_name = 'Heston_vol.npy'

    if not check_if_spot_is_ready:
        S, V = spot_heston_mc(6 * 30, *params, S0=spot, r=0, M=100, steps=24)
        np.save(spot_name, S)
        np.save(vol_name, V)

    S = np.load(spot_name)
    V = np.load(vol_name)

    matrices = get_product_matrices(payment_percent, call_percent, S, V, r=r,
                                    days=6 * 30,
                                    steps=24,
                                    vol_method=vol_method)

    plot_and_write_product_prices(*matrices, plot_prices=plot_prices, plot_vol=plot_vol)


def return_true_false_from_y_n(letter):
    if letter == 'y':
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility product')
    parser.add_argument('spot', type=float, help='Spot')
    parser.add_argument('k', type=float, help='Strike percent')
    parser.add_argument('p', type=float, help='Payment percent (0..1)')
    parser.add_argument('--v', type=str, default='Heston_vol',
                        help='Method for volatility counting ("Historical_vol" for historical volatility,'
                             ' "Heston_vol" for Heston volatility)')
    parser.add_argument('--p1', type=str, default='n', help='Plot prices (cumulative prices)y/n')
    parser.add_argument('--p2', type=str, default='n', help='Plot prices and volatility (real prices) y/n')
    parser.add_argument('--s', type=str, default='n', help='Files "Heston_spot.npy", "Heston_vol.npy" are ready y/n')
    parser.add_argument('--r', type=float, default=0, help='Risk free rate')
    args = parser.parse_args()

    get_product_price(payment_percent=args.p,
                      call_percent=args.k,
                      spot=args.spot,
                      r=args.r,
                      vol_method='Heston_vol',
                      check_if_spot_is_ready=return_true_false_from_y_n(args.s),
                      plot_prices=return_true_false_from_y_n(args.p1),
                      plot_vol=return_true_false_from_y_n(args.p2))
