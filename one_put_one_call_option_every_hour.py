import csv
import sys
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import glob as gb

from lib.exchange_data_reader_historical_new_format_backup import HistoricalReaderNewFormat
from lib.exchange_data_reader_historical import HistoricalReader
# from lib.forward_price_counter import ForwardPriceCounter
# from lib.volatility_surface import Volatility
from lib.option import Option
from lib.useful_things import *
from lib.surface_creation import surface_object_from_file, surface_to_file
import lib.option_formulas as opt
from lib.option import get_years_before_expiration
import lib.plotter as pl
from lib.volatility import get_strike_by_delta, delta_slice_for_new_time, interpolate_surface
from lib.surface_creation import get_data_by_reader, get_surface, get_data_by_small_reader
from lib.option_formulas import price_by_BS, vega, gamma
from lib.exchange_data_reader import ExchangeDataReader
from lib.small_reader import ReaderCutByDates


def get_strike_for_one_request_time(sabr, time):
    new_delta_c = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_c'],
                                           time)
    new_delta_p = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_p'],
                                           time)

    strike_c = get_strike_by_delta(sabr.strike_prices, new_delta_c, 0.25)
    strike_p = get_strike_by_delta(sabr.strike_prices, new_delta_p, -0.25)
    return strike_c, strike_p


def get_vol_by_reader(reader, spot, time_before_expiration, delta=0.5, option_type='CALL'):
    # index = np.where(np.array(reader.today) == today)[0][0]
    print(reader.time_before_expiration)
    data = get_data_by_small_reader(reader)

    vol_obj, iv_list = get_surface(data)
    try:
        if delta == 0.5:
            return vol_obj.interpolate_surface(time_before_expiration, spot)
        elif delta == 0.25 and option_type == 'CALL':
            return vol_obj.get_vol_by_time_delta(time_before_expiration, delta, 'CALL')
        elif delta == 0.25 and option_type == 'PUT':
            return vol_obj.get_vol_by_time_delta(time_before_expiration, delta, 'PUT')
        else:
            print('ERROR: incorrect delta or option type')
            sys.exit(1)
    except:
        return np.nan


def count_prices_for_one_put_one_call_options(reader_list, date_for_file_name, check_vol):
    start_hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'] + np.arange(10, 24).astype(
        'str').tolist()
    print(start_hours)
    print(len(start_hours))
    if check_vol:
        suffix = 'every_hour_vol'
    else:
        suffix = 'every_minute_vol'
    for start_hour in start_hours:
        df = pd.DataFrame()
        # df['time_in_minutes'] = np.arange(60)[::-1]
        df['spot_for_one_hour'] = np.nan
        df['time_for_one_hour'] = np.nan
        df['time_before_expiration'] = np.nan
        df['vol_by_spot'] = np.nan
        df['expiration_time'] = np.nan

        spot_for_one_hour, time_for_one_hour, time_before_expiration, vol_by_spot, expiration_time = [], [], [], [], []
        for reader in reader_list:
            s = reader.spot
            t = reader.today
            if t[-8:-6] == start_hour:
                if start_hour != 23:
                    expiration_time.append(t[:-8] + str(int(start_hour) + 1) + ':00:00')
                else:
                    expiration_time.append(t[:-10] + str(int(t[:-10:-8]) + 1) + '00:00:00')
                spot_for_one_hour.append(s)
                time_for_one_hour.append(t)
                time_before_expiration.append(get_years_before_expiration(t, expiration_time[-1]))
                # vol_by_spot.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
                if check_vol:
                    try:
                        vol_by_spot.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
                    except:
                        if len(vol_by_spot) > 0:
                            prev_vol = vol_by_spot[-1]
                            vol_by_spot.append(prev_vol)
                        else:
                            vol_by_spot.append(np.nan)
                else:
                    if len(spot_for_one_hour) == 1:
                        vol_by_spot.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
                    else:
                        prev_vol = vol_by_spot[0]
                        vol_by_spot.append(prev_vol)
        if len(spot_for_one_hour) != 0:
            df['spot_for_one_hour'] = spot_for_one_hour
            df['time_for_one_hour'] = time_for_one_hour
            df['time_before_expiration'] = time_before_expiration
            df['expiration_time'] = expiration_time
            df['vol_by_spot'] = vol_by_spot
            try:
                print(type(df.spot_for_one_hour.values[0]), type(df.time_before_expiration.values[0]),
                      type(df.vol_by_spot.values[0]))
                df['call_price'] = [price_by_BS(s, s, t, v, 'CALL') for s, t, v in
                                    zip(df.spot_for_one_hour.values, df.time_before_expiration.values,
                                        df.vol_by_spot.values)]
                df['put_price'] = [price_by_BS(s, s, t, v, 'PUT') for s, t, v in
                                   zip(df.spot_for_one_hour.values, df.time_before_expiration.values,
                                       df.vol_by_spot.values)]
                df['sum_price'] = df.call_price + df.put_price
                print()
                print('*' * 25)
                print(start_hour)
                print(df)
            except IndexError:
                print(df)
                print('Start hour', start_hour)
            df.to_csv(
                f'one_call_one_put_prices\\call_put_prices_for_{date_for_file_name}_for_start_hour_{start_hour}_{suffix}.csv',
                index=False)


def count_prices_for_3options(reader_list, date_for_file_name, check_vol):
    suffix = 'every_hour_vol'
    # for start_hour in start_hours:
    df = pd.DataFrame()
    df['time_before_expiration'] = np.nan
    df['expiration_time'] = np.nan

    spot_for_one_hour, time_for_one_hour, time_before_expiration, expiration_time = [], [], [], []
    vol_5d, vol_25d_call, vol_25d_put = [], [], []
    for reader in reader_list:
        s = reader.spot
        t = reader.today

        datetime_today = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        datetime_friday = datetime_today + datetime.timedelta((4 - datetime_today.weekday()) % 7)
        datetime_friday = datetime_friday.replace(hour=8, minute=0, second=0)
        expiration_time.append(datetime_friday)
        # else:
        #     expiration_time.append(t[:-10] + str(int(t[:-10:-8]) + 1) + '00:00:00')
        spot_for_one_hour.append(s)
        time_for_one_hour.append(t)
        time_before_expiration.append(get_years_before_expiration(t, expiration_time[-1]))
        # vol_by_spot.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
        if check_vol:
            try:
                # TODO
                # ДОБАВИТЬ СЮДА ЕЩЕ 2 ВОЛАТИЛЬНОСТИ ДЛЯ РАЗНЫХ ДЕЛЬТ И ВОТКНУТЬ ИХ В ДАТАФРЕЙМ
                # ПРОВЕРИТЬ, ЧТО В ФАЙЛ РЕАЛЬНО ВЫВОДЯТСЯ ДАННЫЕ ДЛЯ ТАБЛИЧЕК РАЗ В ЧАС
                # ДОБАВИТЬ ТУДА ЖЕ ВЕГИ ДЛЯ ЭТИХ РАЗНЫХ ТРЕХ ОПЦИОНОВ
                # СКАЗАТЬ, ЧТО ХЭДЖИРОВАНИЕ ПУСТЬ САМИ ДЕЛАЮТ ИЗ ВЕГ, ДАЛЬТ, ЦЕН ОПЦИОНОВ И ИХ ВОЛАТИЛЬНОСТЕЙ.
                vol_5d.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
                vol_25d_call.append(get_vol_by_reader(reader, s, time_before_expiration[-1],
                                                      delta=0.25, option_type='CALL'))
                vol_25d_put.append(get_vol_by_reader(reader, s, time_before_expiration[-1]),
                                   delta=0.25, option_type='PUT')
            except:
                if len(vol_5d) > 0:
                    prev_vol = vol_5d[-1]
                    vol_5d.append(prev_vol)
                    prev_vol = vol_25d_call[-1]
                    vol_25d_call.append(prev_vol)
                    prev_vol = vol_25d_put[-1]
                    vol_25d_put.append(prev_vol)
                else:
                    vol_5d.append(np.nan)
                    vol_25d_call.append(np.nan)
                    vol_25d_put.append(np.nan)
        else:
            if len(spot_for_one_hour) == 1:
                vol_5d.append(get_vol_by_reader(reader, s, time_before_expiration[-1]))
                vol_25d_call.append(get_vol_by_reader(reader, s, time_before_expiration[-1],
                                                      delta=0.25, option_type='CALL'))
                vol_25d_put.append(get_vol_by_reader(reader, s, time_before_expiration[-1],
                                                     delta=0.25, option_type='PUT'))
            else:
                prev_vol = vol_5d[0]
                vol_5d.append(prev_vol)
                prev_vol = vol_25d_call[0]
                vol_25d_call.append(prev_vol)
                prev_vol = vol_25d_put[0]
                vol_25d_put.append(prev_vol)
    if len(spot_for_one_hour) != 0:
        df['spot'] = spot_for_one_hour
        df['date'] = time_for_one_hour
        df['time_before_expiration'] = time_before_expiration
        df['expiration_time'] = expiration_time
        df['vol_for_50delta'] = vol_5d
        # df['vol_for_25delta_call'] = vol_25d_call
        # df['vol_for_25delta_put'] = vol_25d_put
        df['gamma_for_50delta'] = np.nan
        df['gamma_for_25delta_call'] = np.nan
        df['gamma_for_25delta_put'] = np.nan
        try:
            df['call_price_50delta'] = [price_by_BS(s, s, t, v, 'CALL') for s, t, v in
                                        zip(df.spot.values, df.time_before_expiration.values,
                                            df.vol_for_50delta.values)]
            df['put_price_50delta'] = [price_by_BS(s, s, t, v, 'PUT') for s, t, v in
                                       zip(df.spot.values, df.time_before_expiration.values,
                                           df.vol_for_50delta.values)]
            # df['call_price_25delta'] = [price_by_BS(s, s, t, v, 'CALL') for s, t, v in
            #                             zip(df.spot.values, df.time_before_expiration.values,
            #                                 df.vol_for_25delta_call.values)]
            # df['put_price_25delta'] = [price_by_BS(s, s, t, v, 'PUT') for s, t, v in
            #                            zip(df.spot.values, df.time_before_expiration.values,
            #                                df.vol_for_25delta_put.values)]
            df['gamma_for_50delta'] = np.array([gamma(s, s, t, v) for s, t, v in
                                                zip(df.spot.values, df.time_before_expiration.values,
                                                    df.vol_for_50delta.values)])  # / 100
            # df['gamma_for_25delta_call'] = np.array([gamma(s, s, t, v) for s, t, v in
            #                                zip(df.spot.values, df.time_before_expiration.values,
            #                                    df.vol_for_25delta_call.values)]) #/ 100
            # df['gamma_for_25delta_put'] = np.array([gamma(s, s, t, v) for s, t, v in
            #                               zip(df.spot.values, df.time_before_expiration.values,
            #                                   df.vol_for_25delta_put.values)]) #/ 100
            print()
            print('*' * 25)
            print(df)
        except IndexError:
            print('INDEX ERROR')
            print(df)
        df.to_csv(
            f'one_call_one_put_prices\\call_put_prices_for_{date_for_file_name}_3options.csv',
            index=False)


def count_price_for_put_call_options_one_table(reader):
    spot = reader.spot
    today = datetime.datetime.strptime(reader.today, '%Y-%m-%d %H:%M:%S')
    expiration_time = today + datetime.timedelta(hours=1)

    time_before_expiration = get_years_before_expiration(today, expiration_time)
    vol_by_spot = get_vol_by_reader(reader, spot, time_before_expiration)

    call_price = price_by_BS(spot, spot, time_before_expiration, vol_by_spot, 'CALL')
    put_price = price_by_BS(spot, spot, time_before_expiration, vol_by_spot, 'PUT')
    # sum_price = call_price + put_price
    return call_price, put_price, vol_by_spot


def get_vols(reader):
    vols_list = []
    for k in range(len(reader.today)):
        try:
            vols_list.append(get_vol_by_reader(reader, reader.today[k], reader.spot[k], 1 / 24 / 365))
        except:
            if len(vols_list) > 0:
                prev_vol = vols_list[-1]
                vols_list.append(prev_vol)
            else:
                vols_list.append(np.nan)
    return vols_list


def check_date_exist(start_date, end_date, dates_list):
    dates_list = np.array(dates_list)
    dates_list = [str(date)[:-9] for date in dates_list]

    if str(start_date) not in np.str(dates_list) and str(end_date) not in np.str(end_date):
        print('There are no files with the start date and end date in the folder')
        sys.exit(1)
    elif str(start_date) not in np.str(dates_list):
        print('There is no file with the start date in the folder')
        sys.exit(1)
    elif str(end_date) not in np.str(end_date):
        print('There is no file with the end date in the folder')
        sys.exit(1)


def get_list_of_filenames_to_use(start_date, end_date, folder_name):
    file_names_list = gb.glob(folder_name + '\\*_hourly.csv')
    mask_for_file_name = 'OptBestBA_1m_Deribit_BTC_USDT'
    dates_list = []
    for f in file_names_list:
        date = f[len(mask_for_file_name) + len(folder_name) + 2:-4 - len('_hourly')]
        dates_list.append(datetime.datetime.strptime(date, '%Y%m%d'))
    dates_list = np.array(dates_list)
    file_names_list = np.array(file_names_list)
    if len(file_names_list) > 1:
        arr1inds = dates_list.argsort()
        dates_list = dates_list[arr1inds]
        file_names_list = file_names_list[arr1inds]

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    if end_date is None:
        end_date = start_date + datetime.timedelta(hours=1)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    check_date_exist(start_date.date(), end_date.date(), dates_list)

    return dates_list, file_names_list, start_date, end_date


def get_reader_data_for_two_tables():
    call_put_diff = 0.1
    N1, N2 = 0, 30
    # read_row, step = 'first', 1
    # read_row, step = 'last', 1
    read_row, step = 'all', 1

    cut = False
    use_weights = False

    name = '_delta'
    # name = '_spot'

    asset = 'BTC'
    cut_dates_reader = ReaderCutByDates(
        'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\small_one_day_for_call_put_prices.csv')
    cut_dates_reader.read_one_csv()
    df_list = cut_dates_reader.cut_df(datetime.datetime.strptime('2021-05-24 11:00:51', '%Y-%m-%d %H:%M:%S'),
                                      datetime.datetime.strptime('2021-05-24 12:00:51', '%Y-%m-%d %H:%M:%S')
                                      )
    reader_list = []

    for df in df_list:
        reader = ExchangeDataReader('00.csv', 0, 1000, df=df)
        reader.get_data_from_file()
        reader_list.append(reader)

    # print(reader_list)
    return reader_list


def return_true_false_from_y_n(letter):
    if letter == 'y':
        return True
    else:
        return False


# def get_start_end_times(start_date, end_date):
#     start_time =


def main_for_many_files():
    parser = argparse.ArgumentParser(description='Call and put options prices')

    parser.add_argument('start', type=str, help='Start date (for e.g. "2021-05-24 08:00:00")')
    parser.add_argument('--end', type=str, default=None, help='End date (for e.g. "2021-06-24 08:00:00")')
    parser.add_argument('path', type=str, help='Path to folder with data files')
    parser.add_argument('--vol_check', type=str, default='n',
                        help='Count volatility every minute ("y") or every hour ("n")')

    args = parser.parse_args()
    path_to_files = args.path
    start_date = args.start
    end_date = args.end
    check_vol = return_true_false_from_y_n(args.vol_check)

    call_put_diff = 0.1
    N1, N2 = 0, 30
    # read_row, step = 'first', 1
    # read_row, step = 'last', 1
    read_row, step = 'all', 1

    cut = False
    use_weights = False

    name = '_delta'
    # name = '_spot'

    asset = 'BTC'

    dates, file_names, start_date, end_date = get_list_of_filenames_to_use(start_date, end_date, path_to_files)
    print(dates)
    print(file_names)
    for date, file in zip(dates, file_names):
        print(f'Start of counting for {date}')
        cut_dates_reader = ReaderCutByDates(file)
        cut_dates_reader.read_one_csv()
        df_list = cut_dates_reader.cut_df(start_date, end_date)
        reader_list = []
        for df in df_list:
            # print(df)
            reader = ExchangeDataReader('00.csv', 0, 1000, df=df)
            reader.get_data_from_file()
            reader_list.append(reader)

        # count_prices_for_one_put_one_call_options(reader_list, check_vol=check_vol, date_for_file_name=str(date.date()))
        count_prices_for_3options(reader_list, check_vol=check_vol, date_for_file_name=str(date.date()))


if __name__ == '__main__':
    # get_reader_data_for_two_tables()
    main_for_many_files()
