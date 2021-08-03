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

    cut = False
    use_weights = False

    # name = '_spot'
    name = '_delta'

    asset = 'BTC'

    sym = 'a'

    path_new = f'C:/Users/admin/for python/HISTORY/options/{asset}/new_version/'
    path_old = f'C:/Users/admin/for python/HISTORY/options/{asset}/old_version/'

    reader = HistoricalReaderNewFormat(path_new, path_old, 24, f'shp-ohlcv_ftx_{asset}-USDT_h.csv')
    reader.get_data_from_file(call_put_diff, N1, N2, read_row, step)

    spot_list = reader.spot

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
    list_usd_price = []
    list_btc_price = []

    percent = 0.10
    percent_delta = 0.01

    if sym == '':
        percent_call_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        percent_put_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    else:
        percent_call_list = [0.01, 0.01, 0.05, 0.05, 0.10, 0.10]
        percent_put_list = [0.05, 0.10, 0.01, 0.10, 0.01, 0.10]

    btc_list = np.ones_like(percent_call_list)
    usd_list = np.zeros_like(percent_call_list)
    option_list = np.full_like(percent_call_list, None, dtype=object)

    df = pd.DataFrame({'datetime': reader.fake_today, 'spot': reader.spot})
    df['time_check'] = [False if reader.fake_today[n] - reader.today[n] > datetime.timedelta(hours=12) else True for n in range(len(reader.today))]

    for c, p in zip(percent_call_list, percent_put_list):
        df[f'{asset}_c{c}_p{p}{name}'] = np.nan
        df[f'USD_c{c}_p{p}{name}'] = np.nan
        df[f'opt_type_c{c}_p{p}{name}'] = np.nan

    for n in range(start_n, len(reader.today)):
        if not df['time_check'].iloc[n]:
            pprint(f'Reader today != reader fake today!\n{reader.today[n]}, {reader.fake_today[n]}', 'g')
        else:
            # try:
            k = k0 + n

            today = reader.today[k]
            expiration = reader.fake_today[k] + timedelta(hours=24)
            time = get_years_before_expiration(today, expiration)
            spot = reader.spot[k]

            data = get_data_by_reader(reader, k)
            vol_obj, _ = get_surface(data)

            print('-' * 50)
            print(f'{n}/{len(reader.today)}: today: {today}, exp: {expiration}')

            print(f'BEFORE:\n{asset} {np.round(btc_list, 2)}\nusd {np.round(usd_list, 2)}')

            for y in range(len(percent_call_list)):
                percent_call, percent_put = percent_call_list[y], percent_put_list[y]

                if name == '_delta':
                    strike_put = vol_obj.get_strike_by_time_delta(time, 0.5 - percent_call)
                    strike_call = vol_obj.get_strike_by_time_delta(time, 0.5 + percent_put)
                else:
                    strike_put = spot * (1 - percent_put)
                    strike_call = spot * (1 + percent_call)

                vol_call = vol_obj.interpolate_surface(time, strike_call)
                vol_put = vol_obj.interpolate_surface(time, strike_put)

                # pprint(f'percents: {percent_call}, {percent_put}', 'c')
                # pprint(f'strikes: {strike_call}, {strike_put}', 'c')
                # pprint(f'vols: {vol_call}, {vol_put}', 'c')
                try:
                    time_check = df['time_check'].iloc[n + 1]
                except IndexError:
                    time_check = n == len(df)

                if n == start_n:
                    option_list[y] = Option(today, expiration, spot, strike_call, OptionType.call)
                    usd_list[y] = option_list[y].price(vol_call)
                else:
                    if option_list[y].option_type == 'CALL' and option_list[y].strike_price < spot:
                        if df['time_check'].iloc[n - 1]:
                            usd_list[y] += option_list[y].strike_price * btc_list[y]
                            btc_list[y] = 0
                        if time_check:
                            option_list[y] = Option(today, expiration, spot, strike_put, OptionType.put)
                            usd_list[y] += option_list[y].price(vol_put) * usd_list[y] / option_list[y].strike_price

                    elif option_list[y].option_type == 'CALL' and option_list[y].strike_price >= spot:
                        if df['time_check'].iloc[n - 1]:
                            btc_list[y] += usd_list[y] / option_list[y].strike_price
                            usd_list[y] = 0
                        if time_check:
                            option_list[y] = Option(today, expiration, spot, strike_call, OptionType.call)
                            usd_list[y] += option_list[y].price(vol_call) * btc_list[y]

                    elif option_list[y].option_type == 'PUT' and option_list[y].strike_price > spot:
                        if df['time_check'].iloc[n - 1]:
                            btc_list[y] += usd_list[y] / option_list[y].strike_price
                            usd_list[y] = 0
                        if time_check:
                            option_list[y] = Option(today, expiration, spot, strike_call, OptionType.call)
                            usd_list[y] += option_list[y].price(vol_call) * btc_list[y]

                    elif option_list[y].option_type == 'PUT' and option_list[y].strike_price <= spot:
                        if time_check:
                            option_list[y] = Option(today, expiration, spot, strike_put, OptionType.put)
                            usd_list[y] += option_list[y].price(vol_put) * usd_list[y] / option_list[y].strike_price

                df[f'{asset}_c{percent_call}_p{percent_put}{name}'].iloc[k] = btc_list[y]
                df[f'USD_c{percent_call}_p{percent_put}{name}'].iloc[k] = usd_list[y]
                df[f'opt_type_c{percent_call}_p{percent_put}{name}'].iloc[k] = option_list[y].option_type

            print(f'AFTER:\n{asset} {np.round(btc_list, 2)}\nusd {np.round(usd_list, 2)}')

            # except Exception as e:
            #     print(e)
            #     break

    df.to_csv(f'./wheel_results/{asset}_wheel_result_{sym}symmetry{name}.csv', index=False)

    print(df)


if __name__ == '__main__':
    main()
