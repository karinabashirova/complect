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


def hestonMC(spot, time, variance, rho, r, kappa, lmbda, theta, sigma, N=30, M=1000):
    print(
        f'spot {spot}, time {time}, variance {variance}, rho {rho}, kappa {kappa}, lmbda {lmbda}, theta {theta}, sigma {sigma}')

    S = np.full((M, N), spot)

    delta_t = time / N

    for j in range(M):
        St = spot
        vt = variance

        lnSt = np.log(St)
        lnvt = np.log(vt)

        # first = True
        for n in range(1, N):

            e = norm.ppf(np.random.random())
            eS = norm.ppf(np.random.random())
            ev = rho * eS + np.sqrt(1 - rho ** 2) * e

            # prev_lns = lnSt
            # prev_lnv = lnvt
            # prev_v = vt
            # prev = [1 / vt, (kappa + lmbda) * (kappa * theta / (kappa + lmbda) - vt),
            #         0.5 * sigma ** 2 * delta_t, sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev]

            lnSt += (r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * eS
            lnvt += (1 / vt) * ((kappa + lmbda) * (kappa * theta / (kappa + lmbda) - vt) -
                                0.5 * sigma ** 2) * delta_t + sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev

            # if first and (np.isnan(lnvt) or np.isnan(lnSt)):
            #     first = False
            #     print(n)
            #     print(e, eS, ev)
            #     print(prev_lns, prev_lnv)
            #     print(lnSt, lnvt)
            #     print(vt)
            #     print(prev_v)
            #     print(prev)
            #     print()
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

    # for m in range(M):
    #     plt.plot(S[m], color='powderblue')
    # plt.plot(np.mean(S, axis=0), color='black')
    # plt.show()
    # print(len(np.nanmean(S, axis=0)))

    return np.nanmean(S, axis=0)


def classicMC(spot, time, volatility, N=30, M=1000):
    S = np.full((M, N), spot)

    delta_t = time / N

    for j in range(M):
        for n in range(N - 1):
            S[j][n + 1] = S[j][n] * np.exp(
                (- 0.5 * volatility ** 2) * delta_t + volatility * delta_t ** 0.5 * np.random.normal(0, 1))

    # for m in range(M):
    #     plt.plot(S[m], color='powderblue')
    # plt.plot(np.mean(S, axis=0), color='black')
    # plt.show()
    # print(len(np.nanmean(S, axis=0)))
    return np.mean(S, axis=0)


def objective_function(params, real_spot, variance, rho, theta, sigma):
    spot_H = hestonMC(spot=real_spot[0], time=30 / 365, variance=variance[-1],
                      rho=rho, r=0, kappa=params[0], lmbda=0, theta=theta, sigma=sigma)
    diff = spot_H - real_spot
    return np.nansum(np.array(diff) ** 2)


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
    spot_by_MC = classicMC(real_spot[0], 30 / 365, call_vol)

    # print(f'spot {real_spot[0]}')

    spot_list = spot_df[spot_df.columns[1]].iloc[-28 * 24::24]

    variance = spot_list.rolling(14).var().values / real_spot[0] ** 2
    theta = np.mean(variance[-14:])
    volatility = np.log(spot_list / spot_list.shift(1)).rolling(14).std() * np.sqrt(365)
    sigma = np.log(volatility / volatility.shift(1)).std()  # *np.sqrt(365)
    rho = np.corrcoef(spot_list[14:], volatility[14:])[0][1]

    # plt.scatter(spot_list, volatility)
    # plt.show()

    spot_by_heston_MC = hestonMC(spot=real_spot[0], time=30 / 365, variance=variance[-1],
                                 rho=rho, r=0, kappa=0.1, lmbda=0, theta=theta, sigma=sigma)
    # start_params = [0.1]
    # res = minimize(objective_function, start_params, method='L-BFGS-B',
    #                args=(real_spot, variance, rho, theta, sigma))
    # spot_by_minimize_kappa = hestonMC(spot=real_spot[0], time=30 / 365, variance=variance[-1],
    #                              rho=rho, r=0, kappa=res.x[0], lmbda=0, theta=theta, sigma=sigma)
    # print(res)

    # spot_by_heston_MC = hestonMC(spot=100, time=30 / 365, variance=0.01,
    #                              rho=-0.7, r=0.05, kappa=2, lmbda=0.05, theta=0.01, sigma=0.1)

    # print(spot_by_heston_MC)

    # plt.plot(real_spot, label='real')
    plt.plot(spot_by_MC, label='classic mc')
    plt.plot(spot_by_heston_MC, label='heston mc')
    # plt.plot(spot_by_minimize_kappa, label='spot_by_minimize_kappa')
    plt.legend()
    plt.show()

    put_price = price_by_BS(spot, put_strike, 30 / 365, put_vol, OptionType.put)
    # real_call_prices = np.array(
    #     [price_by_BS(real_spot[0], real_spot[0] * call_percent, 1 / 365, call_vol, OptionType.call)])
    call_prices_by_mc = np.array([])#price_by_BS(spot, call_strike, 1 / 365, call_vol, OptionType.call)])
    call_prices_by_heston_mc = np.array([])#price_by_BS(spot, call_strike, 1 / 365, call_vol, OptionType.call)])

    # print(call_strike, call_vol, call_prices_by_mc)

    for t in range(30):
        # real_call_prices = np.append(real_call_prices,
        #                              price_by_BS(real_spot[t], real_spot[t] * call_percent, 1 / 365, call_vol,
        #                                          OptionType.call))
        call_prices_by_mc = np.append(call_prices_by_mc,
                                      price_by_BS(spot_by_MC[t], spot_by_MC[t] * call_percent, 1 / 365, call_vol,
                                                  OptionType.call))
        call_prices_by_heston_mc = np.append(call_prices_by_heston_mc,
                                             price_by_BS(spot_by_heston_MC[t], spot_by_heston_MC[t] * call_percent,
                                                         1 / 365,
                                                         call_vol,
                                                         OptionType.call))

    # print(f'final price by real spot: {-put_price + np.sum(real_call_prices):.2f}$')
    print(f'final price by MC: {-put_price + np.sum(call_prices_by_mc):.2f}$')
    print(f'final price by Heston: {-put_price + np.sum(call_prices_by_heston_mc):.2f}$')
    pd.DataFrame({
        # 'final price by real spot': -put_price + np.sum(real_call_prices),
                  'final price by MC': [-put_price + np.sum(call_prices_by_mc)],
                  'final price by Heston': [-put_price + np.sum(call_prices_by_heston_mc)]}).to_csv(
        'price_by_MC_and_Heston.csv', index=False)

    # plt.plot(real_call_prices, label='price by real spot')
    plt.plot(call_prices_by_mc, label='price by MC')
    plt.plot(call_prices_by_heston_mc, label='price by Heston MC')
    plt.legend()
    plt.show()


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
        print(f'---{k}{"-"*100}')
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
            hedge_prices = np.append(hedge_prices, call_delta*(reader.spot[k] - reader.spot[k-1]))

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
    main2()
