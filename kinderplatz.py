from datetime import timedelta
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

from lib.last_table_reader import LastTableReader
from lib.option import get_years_before_expiration
from lib.surface_creation import get_data_by_reader, get_surface
from lib import plotter
from lib.useful_things import pprint


def plot_it():
    filename_old = 'no_max_cut_empty_dates_IV3_cut_weights_BTCvol_d50_CALL.csv'
    # filename_new = 'no_max_cut_empty_dates__by_delta50_IV3_ETHvol_d50_CALL.csv'
    filename_new2 = 'new_history_no_max_cut_empty_dates__by_delta50_IV3_BTCvol_d50_CALL.csv'
    df_old = pd.read_csv(
        'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\vols\\' + filename_old)
    # df_new = pd.read_csv(
    #     'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\vols\\' + filename_new)
    df_new2 = pd.read_csv(
        'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\vols\\' + filename_new2)

    fig = go.Figure(layout_title_text=filename_old[-19:-16] + filename_old[-13:-9])

    colors = ['rgb(31, 119, 180)',
              'rgb(255, 127, 14)',
              'rgb(44, 160, 44)',
              'rgb(214, 39, 40)',
              'rgb(148, 103, 189)',
              'rgb(140, 86, 75)',
              'rgb(227, 119, 194)',
              'rgb(127, 127, 127)',
              'rgb(188, 189, 34)',
              'rgb(23, 190, 207)']

    for n, e in enumerate(['1w', '2w', '1m', '2m', '3m', '6m', '9m']):
        fig.add_trace(go.Scatter(x=df_old.Datetime, y=df_old[e], name=e, hovertext=df_old.Datetime, opacity=0.5,
                                 line={'color': colors[n]}))
        # fig.add_trace(
        #     go.Scatter(x=df_new.Datetime, y=df_new[e], name='new' + e, hovertext=df_new.Datetime, mode='markers'))
        fig.add_trace(
            go.Scatter(x=df_new2.Datetime, y=df_new2[e], name='new2 ' + e, hovertext=df_new2.Datetime,
                       mode='lines+markers', marker={'color': colors[n]}))
    # fig.add_trace(
    #     go.Scatter(x=df.Datetime, y=np.log(df['Spot'] / df['Spot'].shift(1)).rolling(30 * 24).std() * np.sqrt(365 * 24),
    #                name='RV', hovertext=df.Datetime))
    fig.show()


def make_one_surface():
    # filename = '10012021.csv'
    filename = 'kinder.csv'
    call_put_diff = 0.1
    N1, N2 = 0, 10000
    # read_row, step = 'first', 1
    read_row, step = 'last', 1
    # read_row, step = 'last', 60

    cut = False
    use_weights = False
    # asset = 'BTC'

    reader_btc = LastTableReader(filename)

    # reader_btc = HistoricalReader(file_path + 'OptBestBA_1m_Deribit_BTC_USDT_20210527.csv')
    reader_btc.get_data_from_file(call_put_diff, N1, N2, step)

    reader = reader_btc

    vol = np.full((5, 7, len(reader.today)), np.nan)
    delta = [0.1, 0.25, 0.5, 0.75, 0.9]
    df_spot = pd.DataFrame({'datetime': reader.today, 'spot': reader.spot})
    option_type = 'CALL'
    for k in range(len(reader.today)):
        today = reader.today[k]
        print('-' * 30)
        print(f'{k}: today: {[today, reader.today[k]]}')

        exp1w = today + timedelta(weeks=1)
        exp2w = today + timedelta(weeks=2)
        exp1m = today + timedelta(days=30)
        exp2m = today + timedelta(days=30 * 2)
        exp3m = today + timedelta(days=30 * 3)
        exp6m = today + timedelta(days=30 * 6)
        exp9m = today + timedelta(days=30 * 9)

        expiration_list = [exp1w, exp2w, exp1m, exp2m, exp3m, exp6m, exp9m]
        time_list = [get_years_before_expiration(today, expiration) for expiration in expiration_list]

        data = get_data_by_reader(reader, k)
        print(data.today, data.spot)
        print(data.strikes)
        print(data.times)
        # print(data.prices)
        print()
        surface_obj, iv_list = get_surface(data, cut, use_weights)

        print(surface_obj.params)

        for j in range(len(delta)):
            for m, time in enumerate(time_list):
                pprint(
                    str([delta[j], expiration_list[m], surface_obj.get_vol_by_time_delta(time, delta[j], option_type)]),
                    'y')

        plotter.plot_surface(surface_obj, iv_list)


def plot_surface_by_points():
    x = np.linspace(-1, 1, 70)
    # Y = np.array([[30, 60, 90, 120, 150, 180, 252, 360]] * 5) / 365
    Y = np.array([[30]*5,
                  [60]*5,
                  [90]*5,
                  [120]*5,
                  [150]*5,
                  [180]*5,
                  [252]*5,
                  [360]*5]) / 365

    X = np.array([[0.51205241, 0.82322236, 1.00307909, 1.05303152, 1.09338898],
                  [0.45169407, 0.80089062, 1.00620474, 1.07958184, 1.13906084],
                  [0.42877633, 0.79095238, 1.00936992, 1.10219195, 1.18019918],
                  [0.42070796, 0.7886176, 1.01256436, 1.12249529, 1.21817794],
                  [0.42029305, 0.78997102, 1.01576343, 1.14228841, 1.25423711],
                  [0.4241645, 0.79339018, 1.01896667, 1.16071179, 1.28895982],
                  [0.44147034, 0.80568382, 1.02659241, 1.20227049, 1.36840737],
                  [0.4712191, 0.82923345, 1.03783184, 1.25980782, 1.48046857]])

    Z = np.array([[0.48881465, 0.18669706, 0.05645742, 0.03401947, 0.02074794],
                  [0.54999333, 0.22047029, 0.07857909, 0.04540372, 0.02592544],
                  [0.57408629, 0.2413895, 0.09487449, 0.05284907, 0.02767245],
                  [0.5837997, 0.25469426, 0.10813082, 0.058425, 0.02806231],
                  [0.58638273, 0.26412368, 0.11948849, 0.06253721, 0.02778209],
                  [0.58519703, 0.27124905, 0.1294602, 0.06609271, 0.02723329],
                  [0.57615324, 0.28324245, 0.14993694, 0.07308538, 0.02606067],
                  [0.56175646, 0.29429688, 0.1752391, 0.08229459, 0.02602775]])
    print(X)

    print(Y)
    layout = go.Layout(width=700, height=700, title_text='Chasing global Minima')
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Blues')], layout=layout)

    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))

    fig.add_scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), mode='markers',
                      marker=dict(size=2, color=X.flatten(),
                                  colorscale='Reds'))
    fig.show()


# plot_it()
# make_one_surface()
# plot_surface_by_points()

import sys
for s in sys.path:
    print(s)