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


def product_hestonMC(spot, time, variance, rho, r, kappa, lmbda, theta, sigma, call_strike=0, put_strike=0, M=1000):
    # print(
    #     f'spot {spot}, time {time}, r {r}',
    #     f'variance {variance}, rho {rho}, kappa {kappa}, lmbda {lmbda}, theta {theta}, sigma {sigma}')

    np.random.seed(42)

    N = 30 * 24

    delta_t = time / N

    S = np.full((M, N + 1), spot, dtype=float)

    intrinsic_call = np.zeros(30)
    intrinsic_put = 0

    c = 1 * spot

    for j in range(M):
        St = spot
        vt = variance

        lnSt = np.log(St)
        lnvt = np.log(vt)

        for n in range(1, N + 1):
            e = norm.ppf(np.random.random())
            eS = norm.ppf(np.random.random())
            ev = rho * eS + np.sqrt(1 - rho ** 2) * e

            lnSt += (r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * eS
            # lnvt += (1 / vt) * (-0.5 * sigma ** 2) * delta_t + sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev
            lnvt += (1 / vt) * ((kappa + lmbda) * (kappa * theta / (kappa + lmbda) - vt) -
                                0.5 * sigma ** 2) * delta_t + sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev

            St = np.exp(lnSt)
            vt = np.exp(lnvt)

            S[j][n] = St

            if n % 24 == 0:
                token_count = c / spot

                c = np.exp(-time * r) * max(0, S[j][n] - call_strike) * token_count
                intrinsic_call[n // 24] += c

            p = np.exp(-time * r) * max(0, put_strike - S[j][-1])
            intrinsic_put += p

    return np.array([intrinsic_call / M, intrinsic_put / M])


def hestonMC(spot, time, variance, rho, r, kappa, lmbda, theta, sigma, call_strike=0, put_strike=0, N=30, M=1000):
    # print(
    #     f'spot {spot}, time {time}, variance {variance}, rho {rho}, kappa {kappa}, lmbda {lmbda}, theta {theta}, sigma {sigma}')

    S = np.full((M, N + 1), spot, dtype=float)

    intrinsic_call = 0
    intrinsic_put = 0

    delta_t = time / N
    np.random.seed(42)
    for j in range(M):
        St = spot
        vt = variance

        lnSt = np.log(St)
        lnvt = np.log(vt)

        first = True

        for n in range(1, N + 1):
            e = norm.ppf(np.random.random())
            eS = norm.ppf(np.random.random())
            ev = rho * eS + np.sqrt(1 - rho ** 2) * e

            # prev_lns = lnSt
            # prev_lnv = lnvt
            # prev_v = vt
            # prev = [1 / vt, (kappa + lmbda) * (kappa * theta / (kappa + lmbda) - vt),
            #         0.5 * sigma ** 2 * delta_t, sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev]

            lnSt += (r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * eS
            # lnvt += (1 / vt) * (-0.5 * sigma ** 2) * delta_t + sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev
            lnvt += (1 / vt) * ((kappa + lmbda) * (kappa * theta / (kappa + lmbda) - vt) -
                                0.5 * sigma ** 2) * delta_t + sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev
            # if first and (np.isnan(lnvt) or np.isnan(lnSt)):
            #     first = False
            #     print(n)
            #     print(e, eS, ev)
            # print(prev_lns, prev_lnv)
            # print(lnSt, lnvt)
            # print(vt)
            # print(prev_v)
            # print(prev)
            # print()
            # if n == 1:
            #     print(St, vt)
            #     print(prev_lns, prev_lnv)
            #     print(lnSt, lnvt)
            #     print((r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * eS)
            #     print((r - 0.5 * vt), delta_t, np.sqrt(vt) * np.sqrt(delta_t) * eS)
            #     print()

            St = np.exp(lnSt)
            vt = np.exp(lnvt)

            S[j][n] = St

        intrinsic_call += np.exp(-time * r) * max(0, S[j][-1] - call_strike)
        intrinsic_put += np.exp(-time * r) * max(0, put_strike - S[j][-1])

    # for m in range(M):
    #     plt.plot(S[m], color='powderblue')
    # plt.plot(np.mean(S, axis=0), color='black')
    # plt.show()

    return np.array([intrinsic_call / M, intrinsic_put / M])


def classicMC(spot, time, volatility, call_strike=0, put_strike=0, r=0, N=30, M=1000):
    print(spot, time, volatility, call_strike, put_strike, r, N, M)
    S = np.full((M, N + 1), spot, dtype=float)

    intrinsic_call = 0
    intrinsic_put = 0

    delta_t = time / N

    for j in range(M):
        for n in range(N):
            S[j][n + 1] = S[j][n] * np.exp(
                (r - 0.5 * volatility ** 2) * delta_t + volatility * np.sqrt(delta_t) * np.random.normal())

        intrinsic_call += np.exp(-time * r) * max(0, S[j][-1] - call_strike)
        intrinsic_put += np.exp(-time * r) * max(0, put_strike - S[j][-1])

    for m in range(M):
        plt.plot(S[m], color='powderblue')
    plt.plot(np.mean(S, axis=0), color='black')
    plt.show()

    return [intrinsic_call / M, intrinsic_put / M]


def objective_function(params, real_spot, variance, rho, theta, sigma):
    spot_H = hestonMC(spot=real_spot[0], time=30 / 365, variance=variance[-1],
                      rho=rho, r=0, kappa=params[0], lmbda=0, theta=theta, sigma=sigma)
    diff = spot_H - real_spot
    return np.nansum(np.array(diff) ** 2)


def hedge(spot_list, price_by_BS_list, put_price, call_percent, call_vol, put_strike, put_vol):
    print('hedge()')
    return_list = []
    C0, P0 = price_by_BS_list[0], put_price
    for i in range(0, len(spot_list) - 24, 24):
        price_by_hedge = 0
        price_by_option_part = 0

        if i > 0:
            intrinsic_value_call = max(0, spot_list[i] - call_strike)
            token_count = intrinsic_value_call / spot_list[i]

        call_strike = spot_list[i] * call_percent

        if i > 0:
            for h, j in enumerate(range(i, i + 24)):
                print(f'Day {i // 24}, hour {h}, {j}')
                s = spot_list[j]

                delta0 = token_count * delta(s, call_strike, 1 / 365 - h / 365 / 24, call_vol, 'CALL')
                delta0 -= delta(s, put_strike, 1 / 365 - h / 365 / 24, put_vol, 'PUT')

                C1, P1 = price_by_BS_list[j], price_by_BS(s, put_strike, 30 / 365 - j / 24 / 365, put_vol, 'PUT')

                hedging_part = delta0 * (spot_list[j] - spot_list[j - 1])
                option_part = (C1 - C0) + (P1 - P0)

                price_by_hedge += hedging_part
                price_by_option_part += option_part

                print('\t(C1 - C0)', (C1 - C0), '(P1 - P0)', (P1 - P0))
                print('\tresult', option_part, hedging_part, option_part - hedging_part)
                print()

                C0, P0 = C1, P1
            r = price_by_option_part - price_by_hedge - max(0, spot_list[i + 24] - call_strike)
            if i == len(spot_list) - 1:
                r += max(0, -spot_list[i + 24] + call_strike)
            return_list.append(r)

    return return_list


def main():
    parser = argparse.ArgumentParser(description='Call options')
    parser.add_argument('surf', help='File with BTC surface')
    parser.add_argument('spot', help='File with history_spot')
    parser.add_argument('call', type=float, help='Percent for call strike')
    parser.add_argument('put', type=float, help='Percent for put strike')

    args = parser.parse_args()
    call_percent = args.call
    put_percent = args.put

    file = args.surf
    reader = LastTableReader(file)
    reader.get_data_from_file(call_put_diff=0.1, N1=0, N2=31, step=1)

    sabr_obj, iv_obj_list = get_surface(get_data_by_reader(reader, 0))

    spot = sabr_obj.spot

    put_strike = spot * put_percent
    put_vol = sabr_obj.interpolate_surface(30 / 365, put_strike)

    call_strike = spot * call_percent
    call_vol = sabr_obj.interpolate_surface(1 / 365, call_strike)

    # call_vol = 1.1168757639916085

    spot_df = pd.read_csv(args.spot, header=None)
    spot_df.set_index(spot_df.columns[1], inplace=True)
    real_spot = spot_df[spot_df.columns[1]].iloc[-14 * 24::24].values

    spot_by_MC, price_by_MC = classicMC(real_spot[0], 30 / 365, call_vol, call_percent, N=30 * 24)
    pd.DataFrame({'min_spot': spot_by_MC[0],
                  'mean_spot': spot_by_MC[1],
                  'max_spot': spot_by_MC[2],
                  'min_spot': price_by_MC[0],
                  'mean_spot': price_by_MC[1],
                  'max_spot': price_by_MC[2]}).to_csv('spot_price_for_MC.csv')

    # print(f'spot {real_spot[0]}')

    spot_list = spot_df[spot_df.columns[1]].iloc[-28 * 24::24]

    variance = spot_list.rolling(14).var().values / real_spot[0] ** 2
    theta = np.mean(variance[-14:])
    volatility = np.log(spot_list / spot_list.shift(1)).rolling(14).std() * np.sqrt(365)
    sigma = np.log(volatility / volatility.shift(1)).std()  # *np.sqrt(365)
    rho = np.corrcoef(spot_list[14:], volatility[14:])[0][1]

    put_price = price_by_BS(spot, put_strike, 30 / 365, put_vol, OptionType.put)
    print(f'final price by MC: {-put_price + np.sum(price_by_MC[1]):.2f}$')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    for i in range(3):
        res = hedge(spot_by_MC[i], price_by_MC[i], put_price, call_strike, call_vol, put_strike, put_vol)
        print(i, np.nancumsum(res))
        fig.add_trace(
            go.Scatter(x=np.linspace(0, len(spot_by_MC[i]), len(res)), y=np.nancumsum(res), name='P&L ' + str(i)),
            row=1, col=1
        )

    for i in range(3):
        fig.add_trace(
            go.Scatter(x=np.arange(len(spot_by_MC[i])), y=spot_by_MC[i], name='spot ' + str(i)),
            row=2, col=1
        )
    for j in range(3):
        fig.add_trace(
            go.Scatter(x=np.arange(len(price_by_MC[j])), y=price_by_MC[j], name='price by spot ' + str(j)),
            row=3, col=1
        )

    fig.show()


def main2():
    reader = HistoricalReaderNewFormat(path_new='C:\\Users\\admin\\for python\\HISTORY\\options\\BTC\\new_version\\',
                                       path_old='C:\\Users\\admin\\for python\\HISTORY\\options\\BTC\\old_version\\')
    reader.get_data_from_file(call_put_diff=0.1, N1=0, N2=31, step=24)

    total_real_call_prices = []
    total_call_prices_by_mc = []
    total_call_prices_by_heston_mc = []
    real_spot_we_use = []
    hedge_prices = np.array([])

    for k in range(28, 85):
        print(f'---{k}{"-" * 100}')
        sabr_obj, iv_obj_list = get_surface(get_data_by_reader(reader, k))
        # plotter.plot_surface(sabr_obj, iv_obj_list)

        spot = sabr_obj.spot
        real_spot_we_use.append(spot)

        put_strike = spot / 2
        put_vol = sabr_obj.interpolate_surface(30 / 365, put_strike)

        call_strike = spot * 1.05
        call_vol = sabr_obj.interpolate_surface(1 / 365, call_strike)

        real_spot = reader.spot[k:k + 30]
        spot_by_MC = classicMC(real_spot[0], 30 / 365, call_vol)

        print(f'spot {real_spot[0]}')

        spot_list = pd.Series(reader.spot[k - 28:k])  # spot_df[spot_df.columns[0]].loc[:'2021-Apr-23 00:00:00':24]
        variance = spot_list.rolling(14).var() / real_spot[0] ** 2
        theta = np.mean(variance)
        volatility = pd.Series(np.log(spot_list / spot_list.shift(1)).rolling(14).std() * np.sqrt(365))
        sigma = np.log(volatility / volatility.shift(1)).std()  # *np.sqrt(365)
        print(sigma)
        sigma = np.log(volatility / volatility.shift(1)).std()*np.sqrt(365)
        print(sigma)
        rho = np.corrcoef(spot_list.values[14:], volatility.values[14:])[0][1]

        spot_by_heston_MC = hestonMC(spot=real_spot[0], time=30 / 365, variance=variance.iloc[-1],
                                     rho=rho, r=0, kappa=0.1, lmbda=0, theta=theta, sigma=sigma)

        print(spot_by_heston_MC)

        # plt.plot(real_spot, label='real')
        # plt.plot(spot_by_MC, label='classic mc')
        # plt.plot(spot_by_heston_MC, label='heston mc')
        # # plt.plot(spot_by_minimize_kappa, label='spot_by_minimize_kappa')
        # plt.legend()
        # plt.show()

        put_price = price_by_BS(spot, put_strike, 30 / 365, put_vol, OptionType.put)
        real_call_prices = np.array([])
        call_prices_by_mc = np.array([])
        call_prices_by_heston_mc = np.array([])

        call_delta = 0
        print(call_strike, call_vol, put_strike, put_vol, put_price)

        for t in range(30):
            call_delta += delta(spot_by_MC[t], real_spot[t] * 1.05, 1 / 365, call_vol, OptionType.call)

            real_call_prices = np.append(real_call_prices,
                                         price_by_BS(real_spot[t], real_spot[t] * 1.05, 1 / 365, call_vol,
                                                     OptionType.call))
            call_prices_by_mc = np.append(call_prices_by_mc,
                                          price_by_BS(spot_by_MC[t], spot_by_MC[t] * 1.05, 1 / 365, call_vol,
                                                      OptionType.call))
            call_prices_by_heston_mc = np.append(call_prices_by_heston_mc,
                                                 price_by_BS(spot_by_heston_MC[t], spot_by_heston_MC[t] * 1.05, 1 / 365,
                                                             call_vol,
                                                             OptionType.call))
        if k > 0:
            hedge_prices = np.append(hedge_prices, call_delta * (reader.spot[k] - reader.spot[k - 1]))

        print(f'final price by real spot: {np.sum(real_call_prices):.2f}$, {put_price:.2f}$')
        print(f'final price by MC: {np.sum(call_prices_by_mc):.2f}$, {put_price:.2f}$')
        print(f'final price by Heston: {np.sum(call_prices_by_heston_mc):.2f}$, {put_price:.2f}$')

        total_real_call_prices.append(np.sum(real_call_prices))
        total_call_prices_by_mc.append(np.sum(call_prices_by_mc))
        total_call_prices_by_heston_mc.append(np.sum(call_prices_by_heston_mc))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=np.arange(len(real_spot_we_use)), y=real_spot_we_use, name='spot'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(len(real_spot_we_use)), y=total_real_call_prices, name='price by real spot'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(len(real_spot_we_use)), y=total_call_prices_by_mc, name='price by MC'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(len(real_spot_we_use)), y=total_call_prices_by_heston_mc, name='price by Heston MC'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(len(real_spot_we_use)), y=hedge_prices, name='price by hedge?'),
        row=2, col=1
    )
    fig.show()


if __name__ == '__main__':
    # product_hestonMC()
    main2()
