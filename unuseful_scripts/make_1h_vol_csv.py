from datetime import timedelta

from lib import plotter
from lib.exchange_data_reader_historical_new_format_backup import HistoricalReaderNewFormat
from lib.useful_things import *
from lib.option import get_years_before_expiration
# from lib.volatility import get_strike_by_delta, delta_slice_for_new_time, interpolate_surface
from lib.surface_creation import get_data_by_reader, get_surface
import os.path


def get_strike_for_one_request_time(sabr, time):
    new_delta_c = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_c'],
                                           time)
    new_delta_p = delta_slice_for_new_time(sabr.times_before_expiration, sabr.strike_prices, sabr.delta['surface_p'],
                                           time)

    strike_c = get_strike_by_delta(sabr.strike_prices, new_delta_c, 0.25)
    strike_p = get_strike_by_delta(sabr.strike_prices, new_delta_p, -0.25)
    return strike_c, strike_p


def get_vol_from_surface(option, reader, today):
    index = np.where(np.array(reader.today) == today)[0][0]
    data = get_data_by_reader(reader, index)

    vol, points = get_surface(data)

    vol_short = interpolate_surface(vol.surface,
                                    vol.times_before_expiration,
                                    vol.strike_prices,
                                    option.time_before_expiration,
                                    option.strike_price)
    return vol_short


def make_vols_for_last_table(filename):
    call_put_diff = 0.1
    N1, N2 = 0, 10000
    # read_row, step = 'first', 1
    # read_row, step = 'last', 1
    read_row, step = 'all', 60

    cut = False
    use_weights = False

    asset = 'BTC'
    # asset = 'ETH'

    pprint(asset, 'c')

    path_new = f'C:/Users/admin/for python/HISTORY/options/{asset}/new_version/'
    path_old = f'C:/Users/admin/for python/HISTORY/options/{asset}/old_version/'
    reader = HistoricalReaderNewFormat(path_new, path_old, 1, f'shp-ohlcv_ftx_{asset}-USDT_h.csv')

    reader.get_data_from_file(call_put_diff, N1, N2, read_row)

    vol = np.full((5, 7, len(reader.today)), np.nan)
    delta = [0.1, 0.25, 0.5, 0.75, 0.9]
    option_type = 'CALL'

    for k in range(0, len(reader.today)):
        # print(f'reader.strikes_count[{k}]: {reader.strikes_count[k]}')
        if reader.fake_today[k] - reader.today[k] < datetime.timedelta(hours=1):
            today = reader.fake_today[k]

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
            # print(data.today, data.spot)
            # print(data.strikes)
            # print(data.times)
            # print(data.prices)
            # print()

            try:
                surface_obj, iv_list = get_surface(data, cut, use_weights)
                check = len(surface_obj.expiration_dates) > 0
            except:
                check = False
                #     ValueError:
                # try:
                #     surface_obj, iv_list = get_surface(data, False, use_weights)
                # except ValueError:
                #     surface_obj, iv_list = get_surface(data, False, False)
            # plotter.plot_surface(surface_obj, iv_list)
            if check:
                for j in range(len(delta)):
                    for m, time in enumerate(time_list):
                        vol[j][m][k] = surface_obj.get_vol_by_time_delta(time, delta[j], option_type)

                    d = {'Datetime': [reader.fake_today[k]], 'Real_datetime': [reader.today[k]], 'Spot': [reader.spot[k]],
                         'Expirations_count': len(surface_obj.expiration_dates)}
                    for m, exp_name in enumerate(['1w', '2w', '1m', '2m', '3m', '6m', '9m']):
                        d[exp_name] = [vol[j][m][k]]
                    # if j == 2:
                    #     print('Delta 50', d['1m'][0])
                    df = pd.DataFrame(d)

                    file = f'./vols/new_history_no_max_cut_empty_dates__by_delta50_IV3_{asset}vol_d{delta[j] * 100:.0f}_{option_type}.csv'
                    if os.path.isfile(file):
                        try:
                            df.to_csv(file, mode='a', index=False, header=False)
                        except PermissionError:
                            file = f'./vols/new_history_strikes_cut_empty_dates_by_delta50_perm_error_IV3_{asset}vol_d{delta[j] * 100:.0f}_{option_type}.csv'

                            df.to_csv(file, mode='a', index=False, header=False)
                    else:
                        df.to_csv(file, index=False)


if __name__ == '__main__':
    make_vols_for_last_table("")
