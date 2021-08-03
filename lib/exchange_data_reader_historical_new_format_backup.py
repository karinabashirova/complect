import pandas as pd
import numpy as np
import datetime
import glob as gb
from lib.useful_things import pprint

pd.options.mode.chained_assignment = None


class HistoricalReaderNewFormat:
    def __init__(self, path_new=None, path_old=None):
        sorted_file_names_new = []
        if path_new is not None:
            file_names_new = gb.glob(path_new + '*hourly.csv')
            prefix_new = file_names_new[0][:-12 - len('_hourly')]
            dates_list_new = np.sort(
                [datetime.datetime.strptime(f[-12 - len('_hourly'):-4 - len('_hourly')], '%Y%m%d') for f in
                 file_names_new])
            sorted_file_names_new = (
                [prefix_new + datetime.datetime.strftime(d, '%Y%m%d') + '_hourly.csv' for d in dates_list_new])

        sorted_file_names_old = []
        if path_new is not None:
            file_names_old = gb.glob(path_old + '*.csv')
            prefix_old = file_names_old[0][len(path_old):len(path_old) + 12]
            dates_list_old = np.sort(
                [datetime.datetime.strptime(f[len(path_old) + 12:len(path_old) + 22], '%Y-%m-%d') for f in file_names_old])
            sorted_file_names_old = (
                [path_old + prefix_old + datetime.datetime.strftime(d, '%Y-%m-%d') + '.csv' for d in dates_list_old])

        self.filenames = np.append(sorted_file_names_old, sorted_file_names_new)
        self.mult = np.append(np.array(np.zeros_like(sorted_file_names_old, dtype=int)),
                              np.array(np.ones_like(sorted_file_names_new, dtype=int)))

        self.today = []
        self.fake_today = []

        self.spot = []

        self.expiration_dates = []
        self.time_before_expiration = []
        self.strikes = []
        self.price = {'ask_c': [], 'ask_p': [], 'bid_c': [], 'bid_p': []}

    def get_data_from_file(self, call_put_diff, N1, N2, read_row='all', step=1):
        df_all = pd.DataFrame()

        for n in range(len(self.filenames)):
            # print(self.filenames[n])

            use_cols = ['k', 'bid_c', 'ask_c', 'bid_p', 'ask_p', 'exp', 'q', 'u_price']
            # use_cols = ['k', 'bid_c', 'ask_c', 'bid_p', 'ask_p', 'e', 'q', 's0']

            df = pd.read_csv(self.filenames[n], comment='r', header=None, skiprows=1)
            # print(df.head(3))
            df_names = pd.read_csv(self.filenames[n], nrows=1)
            # print(df_names)
            # print(df_names.head(2))
            # print()
            columns = df_names.columns.values

            if df_names.columns[0] != 'k' or df_names.columns[0] != 'req':
                df = pd.read_csv(self.filenames[n], comment='r', header=None, skiprows=1)
                if self.mult[n] == 1:
                    columns = ['k', 'ask_c', 'ask_p', 'bid_c', 'bid_p', 'exp', 'q', 'u_price']
                else:
                    columns = ['0', 'k', 'ask_c', 'bid_p', 'bid_c', 'ask_p', 'exp', 'q', 'u_price']

                # else:

            df.rename({i: columns[i] for i in range(len(columns))}, axis=1, inplace=True)
            df = df[['k', 'ask_c', 'bid_p', 'bid_c', 'ask_p', 'exp', 'q', 'u_price']]
            # print(len(df))
            df = df[df.k != 'k']
            # print(len(df))
            df['ask_c'] = df['ask_c'].astype('float')
            df['ask_p'] = df['ask_p'].astype('float')
            df['bid_c'] = df['bid_c'].astype('float')
            df['bid_p'] = df['bid_p'].astype('float')
            df['u_price'] = df['u_price'].astype('float')
            if self.mult[n] == 1:
                # print(df)
                try:
                    df['ask_c'] = df['ask_c'].values * df['u_price'].values
                    df['ask_p'] = df['ask_p'].values * df['u_price'].values
                    df['bid_c'] = df['bid_c'].values * df['u_price'].values
                    df['bid_p'] = df['bid_p'].values * df['u_price'].values
                except TypeError:
                    print(df['ask_c'].values)
                    print('TypeError')
            # print(n, self.mult[n], df.columns)
            df_all = df_all.append(df[use_cols])
        df_all.reset_index(inplace=True)

        df_all.to_csv('btc_df_all.csv')

        # df_all = pd.read_csv(
        #     'C:\\Users\\admin\\for python\\Surface for unknown asset with the very good lib\\complect\\df_all.csv')

        # df_all.drop(df_all[df_all['Unnamed: 0'] == 'ENOENT: no such file or directory'].index, inplace=True)
        df_all.dropna(subset=['q'], inplace=True)
        df_all.reset_index(inplace=True, drop=True)

        today_list_without_seconds = [t[:-2] + '00' if t != 'q' else '' for t in df_all['q'].values]
        today_list_first_iteration, today_index_first_iteration = np.unique(today_list_without_seconds,
                                                                            return_index=True)

        today_unique = today_list_first_iteration[today_list_first_iteration != '']
        today_index = today_index_first_iteration[today_list_first_iteration != '']
        today_index[1:] -= 1

        today_unique = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in today_unique]

        zipped = sorted(zip(today_unique, today_index), key=lambda t: t[0])
        today_unique, today_index = zip(*zipped)

        expiry_real_unique = np.unique(df_all.exp.values)
        expiry_real_unique = expiry_real_unique[(expiry_real_unique != 'exp') & (expiry_real_unique != 'e')]
        expiry_real_unique = sorted([datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in expiry_real_unique])

        max_len = len(expiry_real_unique)

        k_min = int(len(today_unique) % step) - 1

        if k_min < 0:
            k_min = 0
        if read_row == 'first':
            step_range = np.arange(1)

        elif read_row == 'last':
            n1 = len(today_unique) - 1
            n2 = len(today_unique)
            step_range = np.arange(n1, n2, step)

        else:
            step_range = []

            # min_date = np.min(today_unique) + datetime.timedelta(hours=1)
            # min_date -= datetime.timedelta(minutes=min_date.minute)
            # max_date = np.max(today_unique) + datetime.timedelta(hours=1)
            # max_date -= datetime.timedelta(minutes=max_date.minute)

            min_today = np.min(today_unique)
            max_today = np.max(today_unique)
            if min_today.hour == 23 and min_today.minute > 30:
                min_today += datetime.timedelta(days=1)
            min_date = min_today + datetime.timedelta(hours=9 - min_today.hour, minutes=-min_today.minute)
            max_date = max_today + datetime.timedelta(hours=9 - max_today.hour, minutes=-max_today.minute)

            print('min_date, max_date:', min_date, max_date)

            needed_datetime_list = np.arange(min_date,
                                             max_date,
                                             datetime.timedelta(hours=step)).astype(datetime.datetime)

            good_datetime_list = needed_datetime_list.copy()
            self.fake_today = needed_datetime_list

            print('good_datetime_list making started')
            today_unique = np.array(today_unique)
            for i in range(len(needed_datetime_list)):
                if not np.isin(needed_datetime_list[i], today_unique).any():
                    diff = today_unique - needed_datetime_list[i]
                    try:
                        min_index = np.argmax(diff[diff <= datetime.timedelta(days=0)])
                    except ValueError:
                        min_index = 0
                    # pprint(f'\nneed: {needed_datetime_list[i]}', 'y')
                    # pprint(f'min_index: {min_index}, rows: [{today_index[min_index]+1}; {today_index[min_index + 1]})', 'y')
                    # pprint(f'have: {today_unique[min_index]}', 'y')

                    df_small = df_all.iloc[today_index[min_index] + 1:today_index[min_index + 1] + 1].reset_index(
                        drop=True)
                    df_small = df_small.drop(df_small[df_small['k'] == 'k'].index).reset_index(drop=True)

                    while len(np.unique(df_small.exp.values)) <= 1:
                        min_index -= 1
                        df_small = df_all.iloc[today_index[min_index] + 1:today_index[min_index + 1] + 1].reset_index(
                            drop=True)
                        df_small = df_small.drop(df_small[df_small['k'] == 'k'].index).reset_index(drop=True)
                        # pprint(f'\tmin_index: {min_index}, rows: [{today_index[min_index] + 1}; {today_index[min_index + 1]})', 'r')
                        # pprint(f'\thave: {today_unique[min_index]}', 'r')

                    good_datetime_list[i] = today_unique[min_index]
                    step_range.append(min_index)
                else:
                    step_range.append(np.argmax(today_unique == needed_datetime_list[i]))

            print('good_datetime_list making ended')

        print('fields filling started')
        for zz, k in enumerate(step_range):
            today = today_unique[k]

            if k == len(today_unique) - 1:
                df_small = df_all.iloc[today_index[k] + 1:].reset_index(drop=True)
            else:
                df_small = df_all.iloc[today_index[k] + 1:today_index[k + 1] + 1].reset_index(drop=True)

            df_small = df_small.drop(df_small[df_small['k'] == 'k'].index).reset_index(drop=True)
            df_small = df_small.replace(0, np.nan)
            # print('df_small')
            # print(df_small)
            expiry_unique, expiry_index = np.unique(df_small.exp.values, return_index=True)
            expiry_unique = expiry_unique[expiry_unique != 'exp']
            expiry_unique = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in expiry_unique]

            zipped = sorted(zip(expiry_index, expiry_unique), key=lambda t: t[0])
            try:
                expiry_index_sorted, expiry_unique = zip(*zipped)
            except ValueError:
                print('Value Error 1', expiry_index, expiry_unique)
            expiry_unique_sorted = sorted(expiry_unique)

            df_list = []

            for i in range(1, len(expiry_unique)):
                df_small_to_df_list = df_small.iloc[expiry_index_sorted[i - 1]:expiry_index_sorted[i]]
                df_list.append(df_small_to_df_list.reset_index(drop=True))

            df_list.append(df_small.iloc[expiry_index_sorted[-1]:].reset_index(drop=True))

            zipped = sorted(zip(expiry_unique, df_list), key=lambda t: t[0])
            try:
                _, df_list = zip(*zipped)
            except ValueError as e:
                print('Value Error 2', e, expiry_unique, df_list[-1], k, today)
            df_list = list(df_list)

            if len(expiry_unique_sorted) < max_len:
                index = np.flatnonzero(np.isin(expiry_real_unique, expiry_unique_sorted))
                for i in range(max_len):
                    if i not in index:
                        expiry_unique_sorted.insert(i, 'nan')
                        df_list.insert(i, np.nan)

            tmp_time_before_expiration = []
            tmp_strikes = []
            tmp_ask_c = []
            tmp_ask_p = []
            tmp_bid_c = []
            tmp_bid_p = []

            for i, expiration_date in enumerate(expiry_unique_sorted):
                if expiration_date != 'nan':
                    diff = expiration_date - today

                    if diff < datetime.timedelta(days=0, seconds=0, minutes=0, hours=0):
                        print("Date is incorrect\n", "Expiration date", expiration_date, "\nToday", today)
                        expiration_date = 'nan'
                        self.expiration_dates[-1][i] = expiration_date
                    # print(df_list[i])
                    for j in range(len(df_list[i])):
                        ask_c = float(df_list[i]['ask_c'][j])
                        ask_p = float(df_list[i]['ask_p'][j])
                        bid_c = float(df_list[i]['bid_c'][j])
                        bid_p = float(df_list[i]['bid_p'][j])

                        strike = float(df_list[i]['k'][j])
                        spot = float(df_list[i]['u_price'][j])

                        if not np.isnan(ask_c) and not np.isnan(bid_p):
                            if abs(ask_c - bid_p + strike - spot) / spot > call_put_diff:
                                df_list[i]['ask_c'][j] = np.nan
                                df_list[i]['ask_p'][j] = np.nan
                                df_list[i]['bid_c'][j] = np.nan
                                df_list[i]['bid_p'][j] = np.nan
                        elif not np.isnan(ask_c) and not np.isnan(ask_p):
                            if abs(ask_c - ask_p + strike - spot) / spot > call_put_diff:
                                df_list[i]['ask_c'][j] = np.nan
                                df_list[i]['ask_p'][j] = np.nan
                        elif not np.isnan(bid_c) and not np.isnan(bid_p):
                            if abs(bid_c - bid_p + strike - spot) / spot > call_put_diff:
                                df_list[i]['bid_c'][j] = np.nan
                                df_list[i]['bid_p'][j] = np.nan

                    bad_rows_indices = []
                    for j in range(len(df_list[i])):
                        ask_c = float(df_list[i]['ask_c'][j])
                        ask_p = float(df_list[i]['ask_p'][j])
                        bid_c = float(df_list[i]['bid_c'][j])
                        bid_p = float(df_list[i]['bid_p'][j])

                        a = not np.isnan(ask_c) and not np.isnan(ask_p)
                        b = not np.isnan(bid_c) and not np.isnan(bid_p)
                        ab = not np.isnan(ask_c) and not np.isnan(bid_p)

                        all_nan = np.isnan(ask_c) and np.isnan(ask_p) and \
                                  np.isnan(bid_c) and np.isnan(bid_p)

                        if all_nan or not (a or b or ab):
                            bad_rows_indices.append(j)

                    df_list[i].drop(bad_rows_indices, inplace=True)
                    df_list[i] = df_list[i].reset_index(drop=True)

                    tmp_time_before_expiration.append(diff.total_seconds() / (365 * 24 * 60 * 60))
                    tmp_strikes.append(df_list[i]['k'].astype('float').values.tolist())
                    tmp_ask_c.append(
                        df_list[i]['ask_c'].astype('float').values.tolist())
                    tmp_ask_p.append(
                        df_list[i]['ask_p'].astype('float').values.tolist())
                    tmp_bid_c.append(
                        df_list[i]['bid_c'].astype('float').values.tolist())
                    tmp_bid_p.append(
                        df_list[i]['bid_p'].astype('float').values.tolist())
                else:
                    tmp_time_before_expiration.append(np.nan)
                    tmp_strikes.append([np.nan])
                    tmp_ask_c.append([np.nan])
                    tmp_ask_p.append([np.nan])
                    tmp_bid_c.append([np.nan])
                    tmp_bid_p.append([np.nan])

            self.today.append(today)
            self.spot.append(float(df_small.u_price.iloc[0]))
            self.expiration_dates.append(expiry_unique_sorted)
            self.time_before_expiration.append(tmp_time_before_expiration)
            self.strikes.append(tmp_strikes)
            self.price['ask_c'].append(tmp_ask_c)
            self.price['ask_p'].append(tmp_ask_p)
            self.price['bid_c'].append(tmp_bid_c)
            self.price['bid_p'].append(tmp_bid_p)

        self.cut_dates(N1, N2)

        print('READER IS READY!')

    def cut_dates(self, N1, N2):
        for k in range(len(self.today)):
            bad_times = []

            for i, time in enumerate(self.time_before_expiration[k]):
                if (N1 / 365 > time or time > N2 / 365) and not np.isnan(time):
                    bad_times.append(time)

            for time in bad_times:
                index = self.time_before_expiration[k].index(time)

                self.expiration_dates[k].pop(index)
                self.time_before_expiration[k].pop(index)
                self.price['ask_c'][k].pop(index)
                self.price['ask_p'][k].pop(index)
                self.price['bid_c'][k].pop(index)
                self.price['bid_p'][k].pop(index)
                self.strikes[k].pop(index)
