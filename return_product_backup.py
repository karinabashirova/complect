import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from lib.option_formulas import price_by_BS
from Heston_model import price_by_heston
from scipy.optimize import minimize, Bounds, least_squares
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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

    #     plt.plot(S[m])
    # plt.plot(np.mean(S, axis=0), c='black')
    # plt.show()

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


def get_product_matrices(payment_percent, call_percent, S, V, r=0, days=6 * 30, steps=24):
    M, N = S.shape

    strikes_matrix = np.full((M, days), np.nan)
    intrinsic_matrix = np.full((M, days), np.nan)
    calculated_vol_matrix = np.full((M, days), np.nan)
    heston_vol_matrix = V[:, ::steps]
    bs_price_matrix_1 = np.full((M, days), np.nan)
    bs_price_matrix_2 = np.full((M, days), np.nan)
    client_money_matrix = np.full((M, days), np.nan)
    tokens_matrix = np.full((M, days), np.nan)

    # params = [-8.98698206e+00, 1.00000000e-03, 4.81155610e+00, -1.63466446e-01, 5.66211348e-01]
    # heston_price_matrix = np.full((M, days), np.nan)

    for m in range(M):
        strikes_matrix[m] = np.array(
            [call_percent * s for s in S[m][:-steps:steps]])
        intrinsic_matrix[m] = np.array(
            [np.exp(-1 / 365 * r) * max(0, s - k) for s, k in zip(S[m][steps::steps], strikes_matrix[m])])

        calculated_vol_matrix[m] = np.array(
            [np.std(np.log(S[m][1:(n + 1) * steps] / S[m][:(n + 1) * steps - 1])) * np.sqrt(steps * 365)
             for n in range(days)])

        bs_price_matrix_1[m] = np.array(
            [price_by_BS(S[m][n * steps], strikes_matrix[m][n], 1 / 365, calculated_vol_matrix[m][n], 'CALL')
             for n in range(days)]
        )

        bs_price_matrix_2[m] = np.array(
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

    return S, strikes_matrix, calculated_vol_matrix, heston_vol_matrix, intrinsic_matrix, bs_price_matrix_1, bs_price_matrix_2, client_money_matrix, tokens_matrix


def plotting(S, strikes_matrix, calculated_vol_matrix, heston_vol_matrix, intrinsic_matrix, bs_price_matrix_1,
             bs_price_matrix_2,
             client_money_matrix, tokens_matrix):
    # bs_price_matrix = bs_price_matrix * tokens_matrix
    # client_price_matrix = client_money_matrix * tokens_matrix
    # intrinsic_price_matrix = intrinsic_matrix * tokens_matrix

    # print('MEAN OPTION PRICES BY INTRINSIC')
    # print(option_price_list)
    # print('\nMEAN OPTION PRICES BY BS')
    # print(np.mean(bs_price_matrix, axis=0))
    # # print(np.mean(heston_price_matrix, axis=0))
    #
    # print('\nMEAN TOKEN COUNTS')
    # print(np.mean(tokens_matrix, axis=0))

    # print(np.sum(np.nanmean(bs_price_matrix * tokens_matrix, axis=0)),
    #       np.sum(np.nanmean(client_money_matrix * tokens_matrix, axis=0)))
    #
    # plt.plot(np.nancumsum(np.mean(intrinsic_matrix * tokens_matrix, axis=0)) -
    #          np.nancumsum(np.mean(client_money_matrix * tokens_matrix, axis=0)), label='profit by MC')
    # plt.plot(np.nancumsum(np.mean(bs_price_matrix * tokens_matrix, axis=0)) -
    #          np.nancumsum(np.mean(client_money_matrix * tokens_matrix, axis=0)), label='profit by BS')
    # plt.legend()
    # plt.grid()
    #
    # # plt.show()
    #
    # # plt.plot(np.mean(bs_price_matrix, axis=0), label='BS')
    # # # plt.plot(np.mean(heston_price_matrix, axis=0), label='Heston')
    # # plt.plot(option_price_list, label='MC')
    # # plt.legend()
    # # plt.show()
    # #
    # # plt.plot(np.cumsum(np.mean(intrinsic_matrix * tokens_matrix, axis=0)), label='intrinsic')
    # # plt.plot(np.cumsum(np.mean(bs_price_matrix * tokens_matrix, axis=0)), label='BS')
    # # plt.plot(np.cumsum(np.mean(client_money_matrix * tokens_matrix, axis=0)), label='client')
    # # plt.legend()
    # # plt.show()
    #
    # # plt.grid()
    # # plt.show()
    #
    # # for m in range(M):
    # #     i = np.argwhere(np.cumsum(intrinsic_price_matrix[m]) < np.cumsum(client_price_matrix[m]))[0][0]
    # #     if i > 0:
    # #         c = 'black'
    # #         plt.plot(client_price_matrix[m][:i], c)
    # #     else:
    # #         plt.plot(client_price_matrix[m], alpha=0.2)
    # # plt.plot(np.mean(client_price_matrix, axis=0), c='red')
    # # plt.show()
    #
    # # plt.plot(np.nancumsum(np.mean(client_money_matrix * tokens_matrix, axis=0)), c='red')
    # fig = go.Figure()
    # for m in range(M):
    #     fig.add_trace(go.Scatter(x=np.arange(len(bs_price_matrix[m])), y=bs_price_matrix[m], name='In ' + str(m),
    #                              line=dict(color=colors[m])))
    #     fig.add_trace(go.Scatter(x=np.arange(len(client_price_matrix[m])), y=client_price_matrix[m],
    #                              name='Client ' + str(m), mode='markers', marker=dict(color=colors[m])))
    #     try:
    #         i = np.argwhere(np.nancumsum(bs_price_matrix[m]) < np.nancumsum(client_price_matrix[m]))[0][0]
    #     except:
    #         i = 0
    #     if i > 0:
    #         c = 'black'
    #         plt.plot(np.arange(i), (bs_price_matrix[m][:i]) - (client_price_matrix[m][:i]), 'red')
    #         plt.plot(np.arange(i, len(bs_price_matrix[m])), (bs_price_matrix[m][i:]) - (client_price_matrix[m][i:]),
    #                  'black', alpha=0.5)
    #
    #         bs_price_matrix[m][i:] = np.nan
    #         client_price_matrix[m][i:] = np.nan
    #         intrinsic_matrix[m][i:] = np.nan
    #         tokens_matrix[m][i:] = np.nan
    #     else:
    #         plt.plot((bs_price_matrix[m]) - (client_price_matrix[m]), alpha=0.1, c='g')
    # fig.show()

    M, days = strikes_matrix.shape

    x = np.arange(days)

    fig = make_subplots(
        rows=2, cols=1,
        # shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[' ', ' '],
        # x_title='Day',
    )

    name = 'Heston vol'
    # name = 'Vol from spot'

    if name == 'Heston vol':
        final_price_matrix = bs_price_matrix_2 * tokens_matrix
    else:
        final_price_matrix = bs_price_matrix_1 * tokens_matrix
    final_client_money_matrix = client_money_matrix * tokens_matrix

    print(np.sum(np.mean(final_price_matrix, axis=0)), np.sum(np.mean(final_client_money_matrix, axis=0)))

    def count_product_price_by_nan():
        bs_price_matrix_1_ = bs_price_matrix_1.copy()
        bs_price_matrix_2_ = bs_price_matrix_2.copy()
        client_money_matrix_ = client_money_matrix.copy()
        tokens_matrix_ = tokens_matrix.copy()
        for m in range(M):
            indices = np.argwhere(np.nancumsum(bs_price_matrix_2_[m]) < np.nancumsum(client_money_matrix_[m]))

            if len(indices) > 0:
                i = indices[0][0]
                bs_price_matrix_2_[m][i:] = np.nan
                client_money_matrix_[m][i:] = np.nan
                tokens_matrix_[m][i:] = np.nan
        if name == 'Heston vol':
            final_price_matrix_ = bs_price_matrix_2_ * tokens_matrix_
        else:
            final_price_matrix_ = bs_price_matrix_1_ * tokens_matrix_
        final_client_money_matrix_ = client_money_matrix_ * tokens_matrix_

        return np.nansum(np.nanmean(final_price_matrix_, axis=0) - np.nanmean(final_client_money_matrix_, axis=0))

    price_nan = count_product_price_by_nan()
    title = f'Product price by {name} ' + str(np.round(
        np.sum(np.mean(final_price_matrix, axis=0) - np.mean(final_client_money_matrix, axis=0)), 2)) \
            + ', nan price ' + str(np.round(price_nan, 2))

    colors = []
    for m in range(M):
        indices = np.argwhere(final_price_matrix[m] < final_client_money_matrix[m])
        if len(indices > 0):
            colors.append('red')
        else:
            colors.append('green')

    colors = px.colors.qualitative.Plotly * 1000

    for m in range(M):
        def plot_dependencies_on_spot_and_volatility():
            # зависимость от спота
            indices1 = S[m][:-1:24].argsort()

            x1 = np.sort(S[m][:-1:24])

            fig.add_trace(
                go.Scatter(x=x1, y=bs_price_matrix_1[m][indices1],
                           name=f'{m} Calc vol price',
                           mode='markers', marker=dict(color=colors[m])),
                row=1, col=1)

            # fig.add_trace(
            #     go.Scatter(
            #         x=x1, y=bs_price_matrix_2[m][indices1],
            #         name=f'{m} Heston vol price',
            #         mode='lines', line=dict(color=colors[m])),
            #     row=1, col=1)

            # зависимость от волатильности
            indices1 = calculated_vol_matrix[m].argsort()
            indices2 = heston_vol_matrix[m][:-1].argsort()

            x1 = np.sort(calculated_vol_matrix[m])
            x2 = np.sort(heston_vol_matrix[m])

            fig.add_trace(
                go.Scatter(x=x1, y=bs_price_matrix_1[m][indices1],
                           name=f'{m} Calc vol price',
                           mode='markers', marker=dict(color=colors[m])),
                row=2, col=1)

            # fig.add_trace(
            #     go.Scatter(
            #         x=x2, y=bs_price_matrix_2[m][indices2],
            #         name=f'{m} Heston vol price',
            #         mode='lines', line=dict(color=colors[m])),
            #     row=2, col=1)

        def plot_option_prices_and_vol():
            fig.add_trace(
                go.Scatter(x=x, y=calculated_vol_matrix[m],
                           name=f'{m} Calc vol',
                           mode='markers', marker=dict(color=colors[m])),
                row=1, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=heston_vol_matrix[m],
                           name=f'{m} Heston vol',
                           mode='lines', line=dict(color=colors[m])),
                row=1, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=bs_price_matrix_1[m],
                           name=f'{m} Calc vol price',
                           mode='markers', marker=dict(color=colors[m])),
                row=2, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=bs_price_matrix_2[m],
                           name=f'{m} Heston vol price',
                           mode='lines', line=dict(color=colors[m])),
                row=2, col=1)

            fig.add_trace(
                go.Scatter(x=x, y=client_money_matrix[m],
                           name=f'{m} Client money',
                           mode='markers+lines', marker=dict(color=colors[m])),
                row=2, col=1)

        # plot_dependencies_on_spot_and_volatility()
        plot_option_prices_and_vol()

    fig.update_layout(title_text=title)

    fig.show()

    def plot_product_prices():
        fig = go.Figure()
        for m in range(M):
            fig.add_trace(
                go.Scatter(x=x, y=np.nancumsum(final_price_matrix[m] - final_client_money_matrix[m]),
                           name=f'{m} Difference',
                           mode='lines', line=dict(color=colors[m]), opacity=0.4))
        fig.add_trace(
            go.Scatter(x=x, y=np.nancumsum(np.mean(final_price_matrix - final_client_money_matrix, axis=0)),
                       name=f'Mean difference',
                       mode='lines', line=dict(color='black')))

        fig.update_layout(title_text=title)

        fig.show()

    plot_product_prices()


def calculations_with_real_spot():
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


def product_price(payment_percent, call_percent, spot, r, volatility):
    np.set_printoptions(suppress=True)

    # np.random.seed(42)

    # params = [-8.98698206e+00, 1.00000000e-03, 14.81155610e+00, -1.63466446e-01, 2.5]
    # params = [-1, 0.47027231759174926, 1.5431205479472159, -0.37306500528562797, 0.24081117378424158]
    params = [0, 0.47027231759174926, 5.5431205479472159, 0.67306500528562797, 1.]
    print(params)
    spot_name = 'Heston_spot_for_rp_100_kappa0_v0_1.npy'
    vol_name = 'Heston_vol_for_rp_100_kappa0_v0_1.npy'
    # S = spot_bs_mc(S0=spot, volatility=volatility, M=50, days=6 * 30, steps=24)
    #
    S, V = spot_heston_mc(6 * 30, *params, S0=spot, r=0, M=100, steps=24)
    np.save(spot_name, S)
    np.save(vol_name, V)

    # np.save('Heston_spot_for_rp_100_real_params1000.npy', S)
    # np.save('Heston_vol_for_rp_100_real_params1000.npy', V)

    S = np.load(spot_name)  # спот на 100 итерациях
    V = np.load(vol_name)  # спот на 100 итерациях

    matrices = get_product_matrices(payment_percent, call_percent, S, V, r=r, days=6 * 30, steps=24)

    plotting(*matrices)

    # cumsum_call_list = np.cumsum(call_list)
    # cumsum_money_list = np.cumsum(money_list)
    # i = np.argwhere(call_list > money_list)[0][0]
    # print(i)
    # plt.plot(call_list)
    # plt.plot(call_list[:i])
    # plt.plot(money_list)
    # plt.plot(money_list[:i])
    # plt.show()

    # plt.plot(cumsum_call_list)
    # plt.plot(cumsum_money_list)
    # plt.show()
    # print(f'\nCALL SUM:     {np.sum(call_list)}')
    # print(f'\nMONEY SUM:    {np.sum(money_list)}')
    # print(f'\nMONEY - CALL: {np.sum(money_list) - np.sum(call_list)}')


if __name__ == '__main__':
    # S = np.load('Heston_spot_for_rp_100_sigma14.npy')  # спот на 100 итерациях

    # product_price(0.05, 0.97816678, 35000, 0.0, 1.)
    # product_price(0.01485076, 0.99275199, 35000, 0.0, 1.)
    product_price(0.01, 1.01, 35000, 0.0, 1.)

    # def objective_function(x):
    #     a = np.array(get_product_price(x[0], x[1], S, r=0, days=6 * 30, vol=vol[-6 * 30::]))
    #     b = np.array([1000 if i < 100 else -1000 for i in range(len(a))])
    #     print(x)
    #     return np.sqrt(np.sum((a - b) ** 2))
    #
    # bounds = ([1e-3, 0.05], [0.9, 2])
    #
    # res = minimize(objective_function, x0=np.array([0.05, 1.1]), bounds=bounds)
    # print(res)
