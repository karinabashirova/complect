import csv
from datetime import datetime
from datetime import timedelta

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
from lib.surface_creation import get_data_by_reader, get_surface


def get_strike_for_one_request_time(sabr, time):
    new_delta_c = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_c'],
                                           time)
    new_delta_p = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_p'],
                                           time)

    strike_c = get_strike_by_delta(sabr.strike_prices, new_delta_c, 0.25)
    strike_p = get_strike_by_delta(sabr.strike_prices, new_delta_p, -0.25)
    return strike_c, strike_p


def get_vol_by_option_and_reader(option, reader, today):
    index = np.where(np.array(reader.today) == today)[0][0]
    data = get_data_by_reader(reader, index)

    vol_obj, iv_list = get_surface(data)

    return vol_obj.interpolate_surface(option.time_before_expiration, option.strike_price)


def main():
    call_put_diff = 0.1
    N1, N2 = 0, 30
    # read_row, step = 'first', 1
    # read_row, step = 'last', 1
    read_row, step = 'all', 60

    cut = True
    use_weights = True

    file_path = 'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\btc_options\\'

    reader_btc = HistoricalReaderNewFormat(file_path, file_path + 'ALL\\', 24)
    # reader_btc = HistoricalReader(file_path + 'OptBestBA_1m_Deribit_BTC_USDT_20210527.csv')
    reader_btc.get_data_from_file(call_put_diff, N1, N2, read_row, step)

    reader = reader_btc

    spot_list = reader_btc.spot

    start_n = 0
    # zero_date = datetime.datetime.strptime('2021-05-27 09:58:00', '%Y-%m-%d %H:%M:%S')
    zero_date = reader.today[0]
    print('START', zero_date)
    # datetime.datetime.strptime('2021-02-05 08:00:00', '%Y-%m-%d %H:%M:%S')  # reader.today[start_n*24]
    k0 = np.where(np.array(reader.today) == zero_date)[0][0]

    sum_hedge_result = 0
    sum_intrinsic_value = 0

    list_spot = []
    list_is_option_in_money = [None]
    list_option_type = ['CALL']
    list_datetime = []


    percent = 0.10
    percent_delta = 0.01
    percent_call_list = [0.01, 0.02]#, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    percent_put_list = [0.01, 0.02]#, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    btc_list = np.ones_like(percent_call_list)
    usd_list = np.zeros_like(percent_call_list)
    print(btc_list)
    print(type(btc_list), len(btc_list))

    print(len(reader.fake_today), len(reader.today), reader.fake_today)
    df = pd.DataFrame({'datetime': reader.fake_today, 'spot': reader.spot})
    for p in percent_call_list:
        df['BTC_' + str(p) + '_spot'] = np.nan
        df['USD_' + str(p) + '_spot'] = np.nan

    for n in range(start_n, len(reader.today)):
        try:
            k = k0 + n

            today = reader.today[k]
            expiration = reader.fake_today[k] + timedelta(hours=24)
            time = get_years_before_expiration(today, expiration)
            spot = reader.spot[k]

            data = get_data_by_reader(reader, k)
            vol_obj, iv_list = get_surface(data)

            # strike_put = spot * (1 - percent)
            # strike_call = spot * (1 + percent)

            print('-' * 50)
            print('-' * 50)
            print(f'{n}/{len(reader.today)}: today: {today}, exp: {expiration}')
            for y, percent_ in enumerate(zip(percent_call_list, percent_put_list)):
                percent_call, percent_put = percent_[0], percent_[1]
                strike_put = spot * (1 - percent_put)
                strike_call = spot * (1 + percent_call)
                print('Y', y, percent_call, percent_put, strike_call, strike_put)
                print(f'Start btc_list[{y}] = {btc_list[y]}')

                # strike_put = vol_obj.get_strike_by_time_delta(time, 0.5 - percent_delta)
                # strike_call = vol_obj.get_strike_by_time_delta(time, 0.5 + percent_delta)

                vol_call = vol_obj.interpolate_surface(time, strike_call)
                vol_put = vol_obj.interpolate_surface(time, strike_put)

                if n == start_n:
                    option = Option(today, expiration, spot, strike_call, OptionType.call)
                    # vol = get_vol_by_option_and_reader(option, reader, today)
                    usd_list[y] = option.price(vol_call)
                else:
                    print('-' * 50)
                    print(f'BEFORE: btc {btc_list[y]}, usd {usd_list[y]:.2f}')

                    if option.option_type == 'CALL' and option.strike_price < spot:
                        usd_list[y] += option.strike_price * btc_list[y]
                        btc_list[y] = 0
                        option = Option(today, expiration, spot, strike_put, OptionType.put)
                        usd_list[y] += option.price(vol_put) * usd_list[y] / option.strike_price

                        list_is_option_in_money.append(True)
                        list_option_type.append(OptionType.put)
                        # print('Продали все BTC => получили $')

                    elif option.option_type == 'CALL' and option.strike_price >= spot:
                        btc_list[y] += usd_list[y] / option.strike_price
                        usd_list[y] = 0
                        option = Option(today, expiration, spot, strike_call, OptionType.call)
                        usd_list[y] += option.price(vol_call) * btc_list[y]

                        list_is_option_in_money.append(False)
                        list_option_type.append(OptionType.call)
                        # print('Продали колл => получили $')

                    elif option.option_type == 'PUT' and option.strike_price > spot:
                        btc_list[y] += usd_list[y] / option.strike_price
                        usd_list[y] = 0
                        option = Option(today, expiration, spot, strike_call, OptionType.call)
                        usd_list[y] += option.price(vol_call) * btc_list[y]

                        list_is_option_in_money.append(True)
                        list_option_type.append(OptionType.call)
                        # print('Купили BTC за все $, Продали колл => получили $')

                    elif option.option_type == 'PUT' and option.strike_price <= spot:
                        option = Option(today, expiration, spot, strike_put, OptionType.put)
                        usd_list[y] += option.price(vol_put) * usd_list[y] / option.strike_price

                        list_is_option_in_money.append(False)
                        list_option_type.append(OptionType.put)
                        # print('Продали put => получили $')

                    print(f'AFTER: btc {btc_list[y]}, usd {usd_list[y]}')

                df['BTC_' + str(percent_call) + '_spot'].iloc[n] = btc_list[y]
                df['USD_' + str(percent_call) + '_spot'].iloc[n] = usd_list[y]
                print(f'End btc_list[{y}] = {btc_list[y]}')

        except Exception as e:
            print(e)
            break

    df.to_csv('./wheel_results/wheel_result_symmetry.csv',
              index=False)

    print(df)


if __name__ == '__main__':
    main()
