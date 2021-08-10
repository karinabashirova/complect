import datetime
import sys

from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from make_1h_vol_csv_last_table import make_vols_for_last_table, get_vol_for_spot_from_surface
from sklearn.linear_model import LinearRegression
import statistics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from pdf_creator import create_report


def get_IV_for_asset(asset='BTC'):
    if asset == 'BTC':
        # df_BTC = make_vols_for_last_table('last_table_BTC.csv', 'BTC', delta=[0.5], spot_to_csv=False)
        print('reader for BTC')
        return get_vol_for_spot_from_surface('last_table_BTC.csv')
    else:
        print('reader for ETH')
        # df_ETH = make_vols_for_last_table('last_table_ETH.csv', 'ETH', delta=[0.5], spot_to_csv=False)
        return get_vol_for_spot_from_surface('last_table_ETH.csv')


def count_RV_for_each_asset(filename_btc, filename_eth, first_vol_date, last_vol_date, days=30, cut_dates=True):
    df_btc = pd.read_csv(filename_btc, usecols=[1, 2], header=None)
    df_eth = pd.read_csv(filename_eth, usecols=[1, 2], header=None)
    df_btc.columns = ['date', 'Spot_btc']
    df_eth.columns = ['date', 'Spot_eth']
    df_btc = df_btc.drop_duplicates(subset=['date'])
    df_eth = df_eth.drop_duplicates(subset=['date'])
    df_btc.reset_index(inplace=True)
    df_eth.reset_index(inplace=True)

    time_ending = '00:00'
    df_btc = df_btc[[df_btc.date[i][-len(time_ending):] == time_ending for i in range(len(df_btc))]]
    df_eth = df_eth[[df_eth['date'][i][-len(time_ending):] == time_ending for i in range(len(df_eth))]]

    df = pd.merge(left=df_eth, right=df_btc, on=['date'], how='left')

    if cut_dates:
        first_vol_date = datetime.datetime.strptime(first_vol_date, '%Y-%m-%d %H:%M:%S')
        last_vol_date = datetime.datetime.strptime(last_vol_date, '%Y-%m-%d %H:%M:%S')
        print(first_vol_date, last_vol_date)
        df_dates = np.array([datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S') for d in df.date])
        idx1 = np.argwhere(df_dates == first_vol_date)
        idx2 = np.argwhere(df_dates == last_vol_date)
        # if idx2[
        try:
            df = df[idx1[0][0]:idx2[0][0]]
        except:
            pass
    df['RV_btc'] = (np.log(df.Spot_btc / df.Spot_btc.shift(1))).rolling(days * 24).std() * np.sqrt(365 * 24)
    df['returns_btc'] = np.log(df.Spot_btc / df.Spot_btc.shift(1))

    df['RV_eth'] = (np.log(df.Spot_eth / df.Spot_eth.shift(1))).rolling(days * 24).std() * np.sqrt(365 * 24)
    df['returns_eth'] = np.log(df.Spot_eth / df.Spot_eth.shift(1))

    return df.RV_btc[days * 24:], df.returns_btc[days * 24:], df.RV_eth[days * 24:], \
           df.returns_eth[days * 24:], df.Spot_btc[days * 24:], df.Spot_eth[days * 24:], df.date[days * 24:]


def quantile_in_RV(RV, iv, asset):
    percents = [.01, .05, .1, .25, .5, .75, .9, .99]
    btc_quantiles = RV.quantile(percents)
    all_quantiles = statistics.quantiles(RV, n=100)
    quantile = 0
    for i in range(1, len(all_quantiles)):
        if all_quantiles[i - 1] < iv < all_quantiles[i]:
            if np.abs(all_quantiles[i - 1] - iv) > np.abs(all_quantiles[i] - iv):
                print(f'IV for {asset} is near quantile {i}%')
                quantile = i
            else:
                print(f'IV for {asset} is near quantile {i - 1}%')
                quantile = i - 1
            break
    btc_hist = plt.hist(RV.values, bins=10, label='RV ' + asset, alpha=0.7, density=True)
    hist_height = np.max(btc_hist[0])

    for q, p in zip(btc_quantiles, percents):
        plt.plot([q] * 2, [0, hist_height], linestyle='dashed', label='quantile ' + str(p * 100) + '%')

    plt.plot([iv] * 2, [0, hist_height], color='black', label='IV ' + asset + ', q =' + str(quantile) + '%')
    plt.legend()
    plt.savefig(f'plots//quantiles_for_{asset}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()


def quantile_in_RV_for_different_rolling_size(spot, datetimes, asset):
    percents = np.array([.1, .25, .5, .75, .9])
    rolling_sizes = np.array([1, 3, 7, 10, 14, 30, 60, 90, 120, 180, 270, 365])
    min_list = np.zeros(len(rolling_sizes))
    max_list = np.zeros(len(rolling_sizes))
    quantiles = np.zeros((len(rolling_sizes), 5))
    iv_list, _ = get_vol_for_spot_from_surface(f'last_table_{asset}.csv', days_before_expiration=rolling_sizes)
    fig = go.Figure()
    for j, days in enumerate(rolling_sizes):
        rv = pd.Series((np.log(spot / spot.shift(1))).rolling(days * 24).std() * np.sqrt(365 * 24))
        fig.add_trace(go.Scattergl(x=np.array([datetime.datetime.strptime(d, '%Y-%b-%d %H:%M:%S') for d in datetimes]),
                                   y=rv, name=f'RV for {days} days'))
        min_list[j] = np.min(rv)
        max_list[j] = np.max(rv)
        quantiles[j] = rv.quantile(percents)
    fig.update_yaxes(title_text="Volatility")
    fig.update_xaxes(title_text="Datetimes")
    fig.update_layout(legend=dict(font=dict(family="Courier", size=30, color="black")))

    pio.write_image(fig, f'plots//volatility_{asset}.png', width=1980 * 1.1, height=1080 * 1.1)
    # fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=rolling_sizes, y=min_list, name='min'))
    for i in range(len(percents)):
        fig.add_trace(go.Scattergl(x=rolling_sizes, y=quantiles.T[i], name='q = ' + str(percents[i])))
    fig.add_trace(go.Scattergl(x=rolling_sizes, y=iv_list, name='IV', mode='lines',
                               line=dict(color='black')))
    for j, days in enumerate(rolling_sizes):
        fig.add_trace(go.Scattergl(x=[days] * 2, y=[min_list[j], quantiles[j][-1]], mode='lines', opacity=0.2, name='',
                                   line=dict(color='black', dash='dash')))

    fig.update_xaxes(title_text="Days count")
    fig.update_yaxes(title_text="Volatility")
    fig.update_layout(title={'text': asset})
    fig.update_layout(legend=dict(font=dict(family="Courier", size=30, color="black")))

    pio.write_image(fig, f'plots//quantile_in_RV_for_different_rolling_size_{asset}.png', width=1980 * 1.1,
                    height=1080 * 1.1)
    fig.add_trace(go.Scattergl(x=rolling_sizes, y=max_list, name='max'))
    for j, days in enumerate(rolling_sizes):
        fig.add_trace(go.Scattergl(x=[days] * 2, y=[min_list[j], max_list[j]], mode='lines', opacity=0.2, name='',
                                   line=dict(color='black', dash='dash')))
    fig.update_layout(legend=dict(font=dict(family="Courier", size=30, color="black")))
    pio.write_image(fig, f'plots//quantile_in_RV_for_different_rolling_size_{asset}_max.png', width=1980 * 1.1,
                    height=1080 * 1.1)
    # fig.show()


def ratio_quantile_in_RV(RV_BTC, iv_BTC, RV_ETH, iv_ETH):
    iv_ratio = iv_BTC / iv_ETH
    rv_ratio = RV_BTC / RV_ETH
    plt.plot(rv_ratio, label='RV(BTC)/RV(ETH)')
    plt.legend()
    plt.savefig(f'plots//ratio_RV.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    percents = [.01, .05, .1, .25, .5, .75, .9, .99]

    rv_quantiles = rv_ratio.quantile(percents)
    q = 0
    all_quantiles = statistics.quantiles(rv_ratio, n=100)
    for i in range(1, len(all_quantiles)):
        if all_quantiles[i - 1] < iv_ratio < all_quantiles[i]:
            if np.abs(all_quantiles[i - 1] - iv_ratio) > np.abs(all_quantiles[i] - iv_ratio):
                print(f'Ratio RV(BTC)/RV(ETH) is near quantile {i}%')
                q = i
            else:
                print(f'Ratio RV(BTC)/RV(ETH) is near quantile {i - 1}%')
                q = i - 1
            break

    btc_hist = plt.hist(rv_ratio.values, bins=10, label='ratio RV(BTC)/RV(ETH), q =' + str(q) + '%', alpha=0.7,
                        density=True)
    hist_height = np.max(btc_hist[0])

    for q, p in zip(rv_quantiles, percents):
        plt.plot([q] * 2, [0, hist_height], linestyle='dashed', label='quantile ' + str(p * 100) + '%')

    plt.plot([iv_ratio] * 2, [0, hist_height], color='black', label='ratio IV(BTC)/IV(ETH)')
    plt.legend()
    plt.savefig(f'plots//quantiles_for_ratio.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()


def beta_from_returns(hv_btc, returns_BTC, hv_eth, returns_ETH, extra_iv_btc, extra_iv_eth, datetimes):
    fig = go.Figure()
    X = returns_BTC.values
    Y = returns_ETH.values

    # hourly_y, hourly_x = [], []
    # for i in range(30, len(X)):
    #     x = X[:i]
    #     y = Y[:i]
    #     try:
    #         z = np.polyfit(x, y, 1)
    #     except TypeError:
    #         print(len(y), len(y))
    #
    #     hourly_y.append(z[0] * x[-1] + z[1])
    #     hourly_x.append(x[-1])
    # plt.plot(x, z[0] * x + z[1])

    Z = np.polyfit(X, Y, 1)  # линейная регрессия по ретернам спота
    print('R2 for real beta', r2_score(Y, Z[0] * X + Z[1]))
    # plt.scatter(X, Y)
    # plt.plot(X, Z[0] * X + Z[1], c='red')
    # plt.show()

    Z_ = np.polyfit(hv_btc, hv_eth, 1)  # линейная регрессия по историческим волатильностям
    print('R2 for my beta', r2_score(hv_eth, Z[0] * hv_btc + Z[1]))

    # plt.scatter(hv_btc, hv_eth)
    # plt.plot(hv_btc, Z_[0] * hv_btc + Z_[1], c='red')
    # plt.show()
    print(f'Beta k = {Z[0]}, c = {Z[1]}')
    # print(f'My beta k = {Z_[0]}, c = {Z_[1]}')
    if len(extra_iv_eth) > 1:
        print('Real ETH IV', extra_iv_eth[0])
        print('ETH IV by beta', Z[0] * extra_iv_btc[0] + Z[1])
        # print('ETH IV by real polyfit', Z_[0] * extra_iv_btc[0] + Z_[1])
        print('Real BTC IV', extra_iv_btc[0])
    else:
        print('Real ETH IV', extra_iv_eth)
        print('ETH IV by beta', Z[0] * extra_iv_btc[0] + Z[1])
        # print('ETH IV by real polyfit', Z_[0] * extra_iv_btc[0] + Z_[1])
        print('Real BTC IV', extra_iv_btc)
    # R = LinearRegression().fit(X.reshape(-1, 1), Y)
    # print(R.coef_)
    # print(R)
    # plt.plot(hv_btc, R.coef_[0] * hv_btc, label='ETH RV by beta from regression')

    fig.add_trace(go.Scatter(x=hv_btc, y=Z[0] * hv_btc + Z[1], name='ETH RV by beta', mode='lines'))
    fig.add_trace(go.Scatter(x=hv_btc, y=Z_[0] * hv_btc + Z_[1], name='ETH RV by real polyfit', mode='lines'))

    indices = np.array(hv_btc).argsort()

    fig.add_trace(go.Scatter(x=np.array(hv_btc)[indices[::-1]], y=np.array(hv_eth)[indices[::-1]],
                             name='real ETH RV (sorted)', mode='markers', opacity=0.5))
    for i, be in enumerate(zip(extra_iv_btc, extra_iv_eth)):
        b, e = be[0], be[1]
        fig.add_trace(go.Scatter(x=[b] * 2, y=[Z[0] * b + Z[1], e], name='', mode='lines',  # name=str(datetimes[i]),
                                 line=dict(dash='dash'), opacity=0.5))  # Z_ если нужна регрессия по HV

    fig.add_trace(go.Scatter(x=extra_iv_btc, y=Z[0] * extra_iv_btc + Z[1], name='ETH IV by beta', mode='markers',
                             marker=dict(size=10)))  # Z_ если нужна регрессия по HV

    fig.add_trace(go.Scatter(x=extra_iv_btc, y=extra_iv_eth,
                             name='real ETH IV', mode='markers', marker=dict(size=10)))

    fig.update_xaxes(title=dict(text='BTC volatility'))
    fig.update_yaxes(title=dict(text='ETH volatility'))

    fig.update_layout(title_text=f'R2 by beta {np.round(r2_score(extra_iv_eth, Z[0] * extra_iv_btc + Z[1]), 3)}',
                      font=dict(size=30))  # Z_ если нужна регрессия по HV

    print('Linear regression of spot returns')
    print('R2 by real beta', r2_score(extra_iv_eth, Z[0] * extra_iv_btc + Z[1]))
    # print('Linear regression of RV')
    # print('R2 by beta', r2_score(extra_iv_eth, Z_[0] * extra_iv_btc + Z_[1]))
    fig.update_layout(legend=dict(font=dict(family="Courier", size=50, color="black")))
    pio.write_image(fig, 'plots//beta_regression.png', width=1980 * 1.1, height=1080 * 1.1)
    # fig.show()


def plot_HV_and_IV_ratios(hv_btc, hv_eth, iv_btc, iv_eth, datetimes):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetimes, y=hv_btc / hv_eth,
                             name='RV(BTC)/RV(ETH)', mode='markers+lines'))

    fig.add_trace(go.Scatter(x=datetimes, y=iv_btc / iv_eth,
                             name='IV(BTC)/IV(ETH)', mode='markers+lines'))

    fig.update_xaxes(title=dict(text='Datetime'))
    fig.update_yaxes(title=dict(text='vol BTC/ vol ETH'))
    fig.write_image('plots//ratios_RV_IV.png')
    # fig.show()


def read_iv_from_files(file_name_btc, file_name_eth):
    iv_btc_list = pd.read_csv(file_name_btc)
    iv_eth_list = pd.read_csv(file_name_eth)
    iv_eth_list = iv_eth_list.drop_duplicates(subset=['Datetime'], keep='last')
    iv_btc_list = iv_btc_list.drop_duplicates(subset=['Datetime'], keep='last')
    iv_eth_list.rename(columns={'1m': '1m_eth'}, inplace=True)
    iv_btc_list.rename(columns={'1m': '1m_btc'}, inplace=True)
    iv_df = pd.merge(iv_eth_list, iv_btc_list, on='Datetime', how='inner')
    return iv_df


def count_something_for_mean_reversion_of_IV_ratio(iv_btc, iv_eth, datetimes):
    iv_ratio = iv_btc / iv_eth
    max_iv = np.max(iv_ratio)
    min_iv = np.min(iv_ratio)
    mean_iv = np.mean(iv_ratio)
    max_coef = 0.8
    min_coef = 1.8
    # max_coef = 0.8
    # min_coef = 1.8
    mean_coef = 0.05
    if max_coef * max_iv < (1 + mean_coef) * mean_iv:
        print(f'Max coef is too small, should be > {(1 + mean_coef) * mean_iv / max_iv}')
        sys.exit(2)
    elif min_coef * min_iv > (1 - mean_coef) * mean_iv:
        print(f'Min coef is too big, should be < {(1 - mean_coef) * mean_iv / min_iv}')
        sys.exit(2)
    mean_reversion = np.zeros(len(iv_ratio))
    mean_reversion_time = np.zeros(len(iv_ratio))
    days_count = 7
    print(f'Real max {max_iv}, {max_iv * max_coef}')
    print(f'Real min {min_iv}, {min_iv * min_coef}')
    print(f'Mean {mean_iv * (1 - mean_coef), mean_iv, mean_iv * (1 + mean_coef)}')
    for i in range(len(iv_ratio) - days_count * 24):
        if iv_ratio[i] > max_iv * max_coef:
            for dt in range(days_count * 24):
                if mean_iv * (1 - mean_coef) < iv_ratio[i + dt] < mean_iv * (1 + mean_coef):
                    # print('Max', dt, i + dt)
                    mean_reversion[i] = iv_ratio[i] - mean_iv
                    mean_reversion_time[i] = dt
                    break
        elif iv_ratio[i] < min_iv * min_coef:
            for dt in range(days_count * 24):
                if mean_iv * (1 - mean_coef) < iv_ratio[i + dt] < mean_iv * (1 + mean_coef):
                    # print('Min', dt, i + dt)
                    mean_reversion[i] = -iv_ratio[i] + mean_iv
                    mean_reversion_time[i] = dt
                    break
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=datetimes, y=iv_ratio, name='IV(BTC/IV(ETH))', mode='lines',
                               line=dict(color='aquamarine')))

    fig.add_trace(go.Scattergl(x=datetimes, y=[max_iv] * len(datetimes), name='max(IV(BTC/IV(ETH)))', mode='lines',
                               line=dict(color='blue')))
    fig.add_trace(go.Scattergl(x=datetimes, y=[max_iv * max_coef] * len(datetimes),
                               name=f'{max_coef}*max(IV(BTC/IV(ETH)))',
                               mode='lines',
                               line=dict(color='blue'),
                               opacity=0.5))

    fig.add_trace(go.Scattergl(x=datetimes, y=[mean_iv * (1 + mean_coef)] * len(datetimes),
                               name=f'{1 + mean_coef}*mean(IV(BTC/IV(ETH)))',
                               mode='lines',
                               line=dict(color='green'),
                               opacity=0.5))
    fig.add_trace(go.Scattergl(x=datetimes, y=[mean_iv] * len(datetimes), name='mean(IV(BTC/IV(ETH)))', mode='lines',
                               line=dict(color='green')))
    fig.add_trace(go.Scattergl(x=datetimes, y=[mean_iv * (1 - mean_coef)] * len(datetimes),
                               name=f'{1 - mean_coef}*mean(IV(BTC/IV(ETH)))',
                               mode='lines',
                               line=dict(color='green'),
                               opacity=0.5))

    fig.add_trace(go.Scattergl(x=datetimes, y=[min_iv * min_coef] * len(datetimes),
                               name=f'{min_coef}*min(IV(BTC/IV(ETH)))',
                               mode='lines',
                               line=dict(color='red'),
                               opacity=0.5))
    fig.add_trace(go.Scattergl(x=datetimes, y=[min_iv] * len(datetimes), name='min(IV(BTC/IV(ETH)))', mode='lines',
                               line=dict(color='red')))
    fig.show()

    fig = make_subplots(rows=3, cols=1, specs=[[{}], [{}], [{}]],
                        subplot_titles=("Mean reversion for 7 days", 'Time before mean reversion (in hours)',
                                        "Cumulative mean reversion"),
                        row_heights=[0.25, 0.25, 0.5]
                        )
    fig.add_trace(go.Scattergl(x=datetimes, y=mean_reversion), row=1, col=1)
    fig.add_trace(go.Scattergl(x=datetimes, y=mean_reversion_time), row=2, col=1)
    fig.add_trace(go.Scattergl(x=datetimes, y=np.cumsum(mean_reversion)), row=3, col=1)
    fig.update_xaxes(title_text="datetime")
    fig.show()


def make_report(report_name='quantiles.pdf'):
    # Вещь для сравнения волатильностей BTC и ETH
    # Сравниваем с целью сделать вывод, хорошо они соотносятся или одна из них слишком дорогая/дешевая
    # по сравнению с другой/ по сравнению с историческим волатильностями

    # волатильность из поверхности (при strike=spot, expiration=1m)
    # это если нужна одна реальная точка (в глубине души их две)
    iv_eth, date_asset = get_IV_for_asset('ETH')
    date_asset = str(date_asset.strftime('%Y-%b-%d %H:%M:%S'))
    iv_btc, _ = get_IV_for_asset('BTC')

    # волатильности из файлов, в которых считали IV на основе исторических данных (из папки vols)
    # если нужно протестировать на куче данных
    # Параметр cut_dates - отрезать данные спота по первой дате из данных волатильности -
    # True увеличивает R^2 на выбранных данных, но нет гарантии, что при экстраполяции реально лучше
    iv = read_iv_from_files('volatility_for_50_delta_BTC.csv', 'volatility_for_50_delta_ETH.csv')
    rv_btc, returns_btc, rv_eth, returns_eth, spot_btc, spot_eth, spot_date = \
        count_RV_for_each_asset('BTCUSDT.csv', 'ETHUSDT.csv', first_vol_date=iv.Datetime.values[0],
                                last_vol_date=iv.Datetime.values[-1],
                                cut_dates=False)

    quantile_in_RV_for_different_rolling_size(spot_btc, spot_date, 'BTC')
    quantile_in_RV_for_different_rolling_size(spot_eth, spot_date, 'ETH')
    # График отношений HV(btc)/HV(eth) и IV(btc)/IV(eth) в зависимости от времени
    # print('*' * 50)
    iv_dates = np.array([datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in iv.Datetime])
    plot_HV_and_IV_ratios(rv_btc.values, rv_eth.values, iv['1m_btc'].values, iv['1m_eth'].values, iv_dates)

    # посмотреть, в какую квантиль попадают IV eth или btc по сравнению с их же историческими волатильностями
    # считается, что если попали в похожие квантили, то радуемся
    # тут используется одна точка волатильности
    print('*' * 50)
    print('Quantiles in real realised volatility')
    quantile_in_RV(rv_btc, iv_btc, 'BTC')
    quantile_in_RV(rv_eth, iv_eth, 'ETH')

    # посмотреть, в какую квантиль попадают отношения IV(eth) / IV(btc) по сравнению
    # с отношением их же исторических волатильностей
    # считается, что если попали в середину гистограммы, то радуемся
    # тут используется одна точка отношения волатильностей
    print()
    print('*' * 50)
    print('Quantiles in ratio of real realised volatility')
    ratio_quantile_in_RV(rv_btc, iv_btc, rv_eth, iv_eth)

    # построили зависимость IV(eth) от IV(btc) при помощи коээфициента линейной регрессии бета
    # (на самом деле коэффициенты из полифита)
    # сказано строить регрессию на ретернах спота, зачем - не ведаю
    # на деле R^2 лучше, если строить регрессию на исторических волатильностях
    # и уже потом считать с помощью этих коэффициентов IV(eth) из IV(btc)
    # сюда по идее тоже можно запихнуть одну точку в формате np.array([iv_btc]), np.array([iv_eth]) и какую-нибудь дату

    # Как эта вещица помогает в оценке волатильностей, я не выяснила
    print()
    print('*' * 50)
    print('IV from beta by linear regression of spot returns')
    rv_btc, returns_btc, rv_eth, returns_eth, spot_btc, spot_eth, spot_date = \
        count_RV_for_each_asset('BTCUSDT.csv', 'ETHUSDT.csv', first_vol_date=iv.Datetime.values[0],
                                last_vol_date=iv.Datetime.values[-1],
                                cut_dates=True)
    beta_from_returns(rv_btc, returns_btc, rv_eth, returns_eth, iv['1m_btc'].values, iv['1m_eth'].values,
                      iv.Datetime.values)

    # beta_from_returns(rv_btc, returns_btc, rv_eth, returns_eth, np.array([iv_btc]), np.array([iv_eth]),
    #                   np.array([date_asset]))

    # Оценка mean reversion поведения
    # count_something_for_mean_reversion_of_IV_ratio(iv['1m_btc'].values, iv['1m_eth'].values, iv.Datetime.values)

    create_report(report_name)


if __name__ == '__main__':
    make_report()
