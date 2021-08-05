import numpy as np
import pandas as pd

from scipy.stats import norm

import plotly.graph_objects as go

import argparse
import sys


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


def spot_heston_mc(days, kappa, theta, sigma, rho, v0, S0, r, M=1000, steps=24 * 12):
    delta_t = 1 / 365 / steps

    N = days * steps + 1

    S = np.full((M, N), S0, dtype=float)
    # S = np.full(N, S0, dtype=float)

    print(f'SPOT CALCULATING --- {M} iterations')
    for m in range(M):
        check_not_finish = True
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
            if not np.isnan(np.sum(S)):
                check_not_finish = False
    print()
    return S


def get_product_price(call_percent, put_percent, S, r, days=30):
    M, N = S.shape
    print(S.shape)
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

    print('\nTOKENS')
    print(tokens / (M - np.mean(nan_count, axis=0)[steps::steps]))

    return intrinsic_call / (M - np.mean(nan_count, axis=0)[steps::steps]), intrinsic_put / (M - np.mean(nan_count))


def product_price(method, spot, r, call_percent, put_percent, volatility=None, params=None, plot=False,
                  ready_spot=False):
    if method == 'BS':
        if volatility is not None:
            S = spot_bs_mc(days=30, volatility=volatility, S0=spot, r=r, M=5000, steps=12)
        else:
            print('Volatility is None')
            sys.exit(2)
    else:
        if params is not None:
            spot_name = 'Heston_spot.npy'
            if ready_spot:
                S = np.load(spot_name)
            else:
                S = spot_heston_mc(30, *params, S0=spot, r=r, M=1000, steps=12)
                np.save(spot_name, S)
                S = np.load(spot_name)

        else:
            print('Heston params vector is None')
            sys.exit(2)

    call_list, put = get_product_price(call_percent, put_percent, S, r=r)
    if plot:
        plot_prices(call_list, put)
    print(f'CALL: {call_list}\nCALL SUM: {np.sum(call_list)}\nPUT: {put}')


def plot_prices(call_prices, put_price):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(len(call_prices)), y=call_prices,
                   name=f'Call prices for 30 one-day options',
                   mode='lines'))
    fig.add_trace(
        go.Scatter(x=np.arange(len(call_prices)), y=[put_price] * len(call_prices),
                   name=f'Put price for one 30-days option',
                   mode='lines'))

    fig.show()


def get_heston_params():
    df = pd.read_csv('heston_params.csv')
    return df.kappa.values[0], df.theta.values[0], df.sigma.values[0], df.rho.values[0], df.v0.values[0]


def return_true_false_from_y_n(letter):
    if letter == 'y':
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protected product')

    parser.add_argument('spot', type=float, help='Spot')
    parser.add_argument('c', type=float, help='Call strike percent')
    parser.add_argument('p', type=float, help='Put strike percent')
    parser.add_argument('m', type=str, help='Method ("BS" / "H")')
    parser.add_argument('--s', type=str, default='n', help='Files "Heston_spot.npy", "Heston_vol.npy" are ready y/n')
    parser.add_argument('--v', type=float, default=None, help='Volatility for BS')
    parser.add_argument('--r', type=float, default=0.0, help='Risk free rate')
    parser.add_argument('--plot', type=str, default='n', help='Plot call and put prices')

    args = parser.parse_args()
    if args.m == 'H':
        params = get_heston_params()
    else:
        params = None

    product_price(
        method=args.m,
        spot=args.spot,
        r=args.r,
        call_percent=args.c,
        put_percent=args.p,
        volatility=args.v,
        params=params, plot=return_true_false_from_y_n(args.plot),
        ready_spot=return_true_false_from_y_n(args.s)
    )
