import numpy as np
import pandas as pd

import argparse
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.stats import norm


def plot_spot_by_MC(S):
    fig = go.Figure()
    for m in range(S.shape[0]):
        fig.add_trace(go.Scatter(x=np.arange(len(S[m])), y=S[m], mode='lines', opacity=0.5, name='S[' + str(m) + ']'))

    fig.add_trace(
        go.Scatter(x=np.arange(len(S[0])), y=np.mean(S, axis=0), mode='lines+markers', line=dict(color='black'),
                   name='Mean spot'))
    fig.update_layout(title='Spot')
    fig.show()


def spot_heston_mc(hours, kappa, theta, sigma, rho, v0, S0, r, M=500, steps=12):
    delta_t = 1 / 24 / 365 / steps

    N = hours * steps + 1

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


def spot_bs_mc(S0, volatility, r=0, hours=1, M=1000, steps=12, echo='off'):
    print(f'SPOT CALCULATING --- {M} iterations')

    dt = 1 / 24 / 365 / steps

    N = hours * steps

    S = np.full((M, N + 1), S0, dtype=float)

    for m in range(M):
        if echo == 'on':
            if m % 1000 == 0:
                print(m, end=' ')

            if m > 0 and m % 10000 == 0 or m == M - 1:
                print()

        Z = np.random.normal(size=N)

        for n in range(N):
            S[m][n + 1] = S[m][n] * np.exp(
                (r - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z[n])

    #     plt.plot(S[m])
    # plt.plot(np.mean(S, axis=0), c='black')
    # plt.show()
    print()
    return S


def get_vp1_value(S0, vol, v1_0, v2_0, p1, method='MC', plot='n'):
    # S0 = 35000
    # vol = 0.6
    # v1_0 = 1000
    # v2_0 = 1000
    #
    # p1 = 0.06

    if method == 'MC':
        name = 'MC'
        print('Spot by standard MC (BS)')
        if vol == -1:
            print('Volatility is empty (for e.g. --vol=0.6)')
            sys.exit(2)
        S = spot_bs_mc(S0, vol, M=10000, echo='off')
        # plot_spot_by_MC(S)
    else:
        name = 'Heston_MC'
        print('Spot by Heston MC')
        params = get_heston_params()
        S, _ = spot_heston_mc(1, *params, S0, 0, M=1000)
        # plot_spot_by_MC(S)

    dv1 = v1_0 * p1
    dv2 = []

    for m in range(S.shape[0]):
        p2 = abs(S[m][-1] / S[m][0] - 1)
        dv2.append(v2_0 * p2)

    dv2 = np.array(dv2)

    final_dv1 = v1_0 - dv1 + dv2
    final_dv2 = v2_0 - dv2 + dv1

    result1 = final_dv1 / v1_0 - 1
    result2 = final_dv2 / v2_0 - 1

    mean1 = np.mean(result1)
    mean2 = np.mean(result2)

    if mean1 >= mean2:
        quantile = np.percentile(result1, 1)
        color = 'blue'
    else:
        quantile = np.percentile(result2, 1)
        color = 'orange'

    v1_quantile_list = []
    v2_quantile_list = []

    for q in [1, 5, 10]:
        v1_quantile_list.append(np.percentile(result1, q))
        v2_quantile_list.append(np.percentile(result2, q))

    print(f'mean for v1: {mean1},\nmean for v2: {mean2},\nquantile for max result: {quantile}')
    pd.DataFrame({
        'v1_mean': [mean1],
        'v1_quantile_1percent': [v1_quantile_list[0]],
        'v1_quantile_5percent': [v1_quantile_list[1]],
        'v1_quantile_10percent': [v1_quantile_list[2]],
        'v2_mean': [mean2],
        'v2_quantile_1percent': [v2_quantile_list[0]],
        'v2_quantile_5percent': [v2_quantile_list[1]],
        'v2_quantile_10percent': [v2_quantile_list[2]]
    }).to_csv(f'volatility_product_results_by_{name}.csv', index=False)

    if plot == 'y':
        bins = 20

        h1 = plt.hist(result1, bins=bins, alpha=0.5)
        plt.hist(result2, bins=bins, alpha=0.5)

        hist_height = np.max(h1[0])

        plt.plot([quantile] * 2, [0, hist_height], color=color, linestyle='dashed')

        plt.show()


def get_heston_params():
    df = pd.read_csv('heston_params.csv')
    return df.kappa.values[0], df.theta.values[0], df.sigma.values[0], df.rho.values[0], df.v0.values[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility product')
    parser.add_argument('spot', type=float, help='Spot')
    parser.add_argument('v1', type=float, help='First count')
    parser.add_argument('v2', type=float, help='Second value')
    parser.add_argument('p1', type=float, help='First percent')
    parser.add_argument('--m', type=str, default='MC',
                        help='Method for spot generation ("MC" for standard MC, "H" for Heston MC)')
    parser.add_argument('--vol', default=-1, type=float, help='Volatility for standard MC')
    parser.add_argument('--p', type=str, default='n', help='Plot а histogram y/n')
    args = parser.parse_args()
    get_vp1_value(args.spot, args.vol, args.v1, args.v2, args.p1, method=args.m, plot=args.p)
