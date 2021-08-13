import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from IV_RV_regression import read_iv_from_files, count_RV_for_each_asset
from lib.option_formulas import vega, price_by_BS, delta


def get_IV_ratio_and_RV():
    iv = read_iv_from_files('volatility_for_50_delta_BTC.csv', 'volatility_for_50_delta_ETH.csv')
    rv_btc, returns_btc, rv_eth, returns_eth, spot_btc, spot_eth, spot_date = \
        count_RV_for_each_asset('BTCUSDT.csv', 'ETHUSDT.csv', first_vol_date=iv.Datetime.values[0],
                                last_vol_date=iv.Datetime.values[-1], cut_dates=True)
    dates = np.array([datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S') for d in spot_date.values])
    iv.Datetime = np.array([datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in iv.Datetime])
    rv_df = pd.DataFrame({"rv_btc": rv_btc.values, 'rv_eth': rv_eth.values, 'Datetime': dates,
                          'spot_btc': spot_btc.values, 'spot_eth': spot_eth.values})
    df = pd.merge(left=iv, right=rv_df, on=['Datetime'], how='left')
    iv_btc = df['1m_btc']
    iv_eth = df['1m_eth']
    # df['iv_ratio'] = iv_btc / iv_eth
    df['iv_ratio'] = iv_eth / iv_btc
    df.rename(columns={'1m_eth': 'iv_eth'}, inplace=True)
    df.rename(columns={'1m_btc': 'iv_btc'}, inplace=True)
    df.rename(columns={'spot_eth_x': 'spot_eth'}, inplace=True)
    df.rename(columns={'spot_btc_x': 'spot_btc'}, inplace=True)
    df = df[['Datetime', 'iv_btc', 'iv_eth', 'iv_ratio', 'rv_btc', 'rv_eth', 'spot_btc', 'spot_eth']]
    df.to_csv('IV_ratio_RV.csv')
    return df


def mean_reversion_of_IV_ratio_different_assets_by_std(df, hedging_function, output, title):
    iv_ratio = df.iv_ratio

    mean_reversion = np.zeros(len(iv_ratio))
    mean_reversion_time = np.zeros(len(iv_ratio))
    days_count = 30  # максимальное число дней для возврата к среднему
    days_count_for_std = 7  # число дней для определения min, max, mean

    std_iv = (iv_ratio.rolling(days_count_for_std * 24).std()).values
    mean_iv = (iv_ratio.rolling(days_count_for_std * 24).mean()).values
    hedging_result = []
    hedging_dates = []
    hedging_dt = []
    for i in range(days_count_for_std * 24, len(iv_ratio) - days_count * 24):
        if iv_ratio[i] > (mean_iv + 2 * std_iv)[i]:  # или > max_iv_real[-1]
            for dt in range(days_count * 24):
                if (mean_iv - std_iv)[i] < iv_ratio[i + dt] < (mean_iv + std_iv)[i]:
                    # if iv_ratio[i + dt] < 1.1*mean_iv[i]:
                    mean_reversion[i] = iv_ratio[i] - mean_iv[i]
                    mean_reversion_time[i] = dt
                    print('dt', dt)
                    if dt > 1:
                        if output:
                            hedging_result.append(hedging_function(df[i - 1:i + dt], dt))
                            hedging_dates.append(df.Datetime.iloc[i])
                            hedging_dt.append(dt)
                        else:
                            hedging_function(df[i - 1:i + dt], dt)
                    # print(df.Datetime.iloc[i], df.Datetime.iloc[i + dt], df.spot_eth.iloc[i],
                    #       df.spot_eth.iloc[i + dt]
                    #       , df.spot_btc.iloc[i], df.spot_btc.iloc[i + dt])
                    break
        elif iv_ratio[i] < (mean_iv - 2 * std_iv)[i]:  # или < min_iv_real[-1]
            for dt in range(days_count * 24):
                if (mean_iv - std_iv)[i] < iv_ratio[i + dt] < (mean_iv + std_iv)[i]:
                    # if mean_iv[i]*0.9 < iv_ratio[i + dt]:
                    mean_reversion[i] = -iv_ratio[i] + mean_iv[i]
                    mean_reversion_time[i] = dt
                    print('dt', dt, mean_reversion_time[i])
                    if dt > 1:
                        if output:
                            hedging_result.append(hedging_function(df[i - 1:i + dt], dt))
                            hedging_dates.append(df.Datetime.iloc[i])
                            hedging_dt.append(dt)
                        else:
                            hedging_function(df[i - 1:i + dt], dt)
                    # print(df.Datetime.iloc[i], df.Datetime.iloc[i + dt], df.spot_eth.iloc[i],
                    #       df.spot_eth.iloc[i + dt]
                    #       , df.spot_btc.iloc[i], df.spot_btc.iloc[i + dt])

                    break
    print(mean_reversion_time)
    if output:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=hedging_dates,
                                   y=np.cumsum(hedging_result), mode='lines+markers',
                                   text=hedging_dt,
                                   name='hedging result'))
        fig.update_layout(title={'text': title})

        fig.show()

    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24],
    #                            y=iv_ratio[days_count_for_std * 24: - days_count * 24],
    #                            name='IV(BTC/IV(ETH))', mode='lines', line=dict(color='aquamarine')))
    #
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24],
    #                            y=(mean_iv + 2 * std_iv), name='mean+2*std(IV(BTC/IV(ETH)))', mode='lines',
    #                            line=dict(color='blue')))
    #
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24], y=(mean_iv + std_iv),
    #                            name=f'{1 + mean_coef}*mean(IV(BTC/IV(ETH)))',
    #                            mode='lines', line=dict(color='green'), opacity=0.5))
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24], y=mean_iv,
    #                            name='mean(IV(BTC/IV(ETH)))', mode='lines', line=dict(color='green')))
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24], y=(mean_iv - std_iv),
    #                            name=f'{1 - mean_coef}*mean(IV(BTC/IV(ETH)))',
    #                            mode='lines', line=dict(color='green'), opacity=0.5))
    #
    # fig.add_trace(go.Scattergl(x=datetimes[days_count_for_std * 24: - days_count * 24], y=(mean_iv - 2 * std_iv),
    #                            name='mean-2*std(IV(BTC/IV(ETH)))', mode='lines', line=dict(color='red')))
    # fig.show()


def vega_hedging(df, N):
    df = df[1:]
    df.reset_index(inplace=True)
    R = df.iv_ratio[0]
    print('*' * 10, 'Vega', '*' * 10)

    print('R', R)
    print(df.Datetime.iloc[0], df.Datetime.iloc[-1], len(df))

    vega_index = 0  # 0 если вега в начальный момент времени
    vega_eth = vega(df.spot_eth.iloc[vega_index], df.spot_eth.iloc[vega_index], 30 / 365, df.iv_eth.iloc[0]) / 100
    vega_btc = vega(df.spot_btc.iloc[vega_index], df.spot_btc.iloc[vega_index], 30 / 365, df.iv_btc.iloc[0]) / 100

    eth_pnl = (df.iv_eth.iloc[0] - df.iv_eth.iloc[-1]) * vega_eth
    btc_pnl = (df.iv_btc.iloc[0] - df.iv_btc.iloc[-1]) * vega_btc * R

    return btc_pnl - eth_pnl


def real_hedging(df, N):
    print('*' * 10, 'Hedging', '*' * 10)
    df.reset_index(inplace=True)
    print(df.Datetime.iloc[1], df.Datetime.iloc[-1], len(df))
    R = df.iv_ratio[1]
    print('R', R)

    # if R > 1:
    #     option_type_btc = 'PUT'
    #     option_type_eth = 'CALL'
    # else:
    #     option_type_btc = 'CALL'
    #     option_type_eth = 'PUT'

    # option_type_btc = 'PUT'
    # option_type_eth = 'CALL'

    # option_type_btc = 'CALL'
    # option_type_eth = 'PUT'

    option_type_btc = 'PUT'
    option_type_eth = 'PUT'

    # option_type_btc = 'CALL'
    # option_type_eth = 'CALL'

    print(f'BTC option type {option_type_btc}, ETH option type {option_type_eth}')
    start_price_btc = price_by_BS(df.spot_btc.iloc[1], df.spot_btc.iloc[1], 30 / 365, df.iv_btc.iloc[1],
                                  option_type=option_type_btc)
    start_price_eth = price_by_BS(df.spot_eth.iloc[1], df.spot_eth.iloc[1], 30 / 365, df.iv_eth.iloc[1],
                                  option_type=option_type_eth)

    end_price_btc = price_by_BS(df.spot_btc.iloc[-1], df.spot_btc.iloc[1], 30 / 365 - N / 24 / 365, df.iv_btc.iloc[-1],
                                option_type=option_type_btc)
    end_price_eth = price_by_BS(df.spot_eth.iloc[-1], df.spot_eth.iloc[1], 30 / 365 - N / 24 / 365, df.iv_eth.iloc[-1],
                                option_type=option_type_eth)
    price_list = []
    for i in range(1, N + 1):
        hours_before_expiration = 30 / 365 - i / 24 / 365
        price_list.append(price_by_BS(df.spot_btc.iloc[i], df.spot_btc.iloc[1], hours_before_expiration,
                                      df.iv_btc.iloc[i], option_type=option_type_btc))
    plt.scatter(df.spot_btc[1:], price_list)
    plt.plot(df.spot_btc[1:], price_list)
    plt.xlabel('spot')
    plt.ylabel('option price')
    plt.show()
    O_btc = R * (end_price_btc - start_price_btc)
    O_eth = (end_price_eth - start_price_eth)
    print('O_btc', O_btc, R, end_price_btc, start_price_btc)

    H_btc = np.zeros(N)
    H_eth = np.zeros(N)

    for i in range(1, N + 1):  # вроде учла предыдущий спот (до покупки опциона)
        hours_before_expiration = 30 / 365 - i / 24 / 365
        delta_eth = delta(df.spot_eth.iloc[i], df.spot_eth.iloc[1], hours_before_expiration, df.iv_eth.iloc[i],
                          option_type=option_type_eth)
        delta_btc = delta(df.spot_btc.iloc[i], df.spot_btc.iloc[1], hours_before_expiration, df.iv_btc.iloc[i],
                          option_type=option_type_btc)

        H_btc[i - 1] = R * delta_btc * (df.spot_btc.iloc[i] - df.spot_btc.iloc[i - 1])
        H_eth[i - 1] = delta_eth * (df.spot_eth.iloc[i] - df.spot_eth.iloc[i - 1])
    btc_part = (O_btc - np.sum(H_btc))
    eth_part = (O_eth - np.sum(H_eth))

    if len(df) > 10:
        plt.title('N =' + str(N) + ', ' + str(df.Datetime.iloc[1]))
        plt.plot(np.cumsum(H_eth - H_btc), label='H_eth - H_btc')
        plt.plot([(O_eth - np.sum(H_eth)) - (O_btc - np.sum(H_btc))] * len(H_btc), label='total result')
        plt.plot([-O_eth] * len(H_btc), label='-O_eth')
        plt.plot([O_btc] * len(H_btc), label='O_btc')
        plt.legend()
        plt.show()

    return btc_part - eth_part


if __name__ == '__main__':
    asset1 = 'BTC'
    asset2 = 'ETH'

    # merge по датам для кучи файлов со спотом и волатильностью
    # df = get_IV_ratio_and_RV()
    df = pd.read_csv('IV_ratio_RV.csv')

    # merge по датам для кучи файлов с волатильностью, тут вроде не подходит
    # iv = read_iv_from_files(f'volatility_for_50_delta_{asset1}.csv', f'volatility_for_50_delta_{asset2}.csv')

    # расчет результатов хэджирования для всех возможных точек + выбор этих самых точек по принципу попадания
    # в два стандартных отклонения, если не попали - начинаем хэджирование/вещь с вегой
    # mean_reversion_of_IV_ratio_different_assets_by_std(df, vega_hedging, output=True, title='vega')
    mean_reversion_of_IV_ratio_different_assets_by_std(df, real_hedging, output=True, title='hedging')
