import pandas as pd
import numpy as np
import datetime
from lib.option import get_years_before_expiration


class LastTableReader:
    def __init__(self, filename):
        self.filename = filename
        self.expiration_dates = []
        self.time_before_expiration = []
        self.spot = []
        self.price = {'ask_c': [], 'ask_p': [], 'bid_c': [], 'bid_p': []}
        self.strikes = []
        self.today = []

    def get_data(self, N1, N2, divided_by_spot=False):
        column_names = ['k', 'ask_c', 'ask_p', 'bid_c', 'bid_p', 'exp', 'q', 'u_price']
        request_datetime_column_index = -2

        df = self.get_last_table_from_file(column_names, request_datetime_column_index)

        df['q'] = df['q'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
        df['exp'] = df['exp'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))

        df.replace(to_replace=0, value=np.nan, inplace=True)

        if divided_by_spot:
            for key in ['ask_c', 'ask_p', 'bid_c', 'bid_p']:
                df[key] = df[key] * df['u_price']

        self.transform_table_to_fields(df, N1, N2)

    def get_last_table_from_file(self, column_names, request_datetime_column_index):
        with open(self.filename) as f:
            r = f.readlines()
            max_row_number = len(r)

            last_datetime = r[-1].split(',')[request_datetime_column_index]
            last_table_row_number = None

            for i in range(max_row_number - 2, 0, -1):
                if r[i].split(',')[request_datetime_column_index] != last_datetime:
                    last_table_row_number = i
                    break

        if last_table_row_number is None:
            last_table_row_number = 0

        df = pd.read_csv(self.filename, skiprows=last_table_row_number + 1, header=None, names=column_names)

        return df

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

    def transform_table_to_fields(self, df, N1, N2):
        self.today = df['q'].iloc[-1]
        self.spot = df['u_price'].iloc[-1]

        expirations = df['exp'].drop_duplicates().sort_values()

        for e in expirations:
            if N1 / 365 < get_years_before_expiration(self.today, e) < N2 / 365:
                small_df = df[df['exp'] == e].reset_index()
                small_df = small_df.sort_values(by='k')

                call_put_diff = 1.1

                for j in range(len(small_df)):
                    strike = float(small_df['k'].iloc[j])
                    spot = float(small_df['u_price'].iloc[j])

                    ask_c = float(small_df['ask_c'].iloc[j])
                    ask_p = float(small_df['ask_p'].iloc[j])
                    bid_c = float(small_df['bid_c'].iloc[j])
                    bid_p = float(small_df['bid_p'].iloc[j])

                    if not np.isnan(ask_c) and not np.isnan(bid_p):
                        if abs(ask_c - bid_p + strike - spot) / spot > call_put_diff:
                            small_df['ask_c'].iloc[j] = np.nan
                            small_df['ask_p'].iloc[j] = np.nan
                            small_df['bid_c'].iloc[j] = np.nan
                            small_df['bid_p'].iloc[j] = np.nan
                    elif not np.isnan(ask_c) and not np.isnan(ask_p):
                        if abs(ask_c - ask_p + strike - spot) / spot > call_put_diff:
                            small_df['ask_c'].iloc[j] = np.nan
                            small_df['ask_p'].iloc[j] = np.nan
                    elif not np.isnan(bid_c) and not np.isnan(bid_p):
                        if abs(bid_c - bid_p + strike - spot) / spot > call_put_diff:
                            small_df['bid_c'].iloc[j] = np.nan
                            small_df['bid_p'].iloc[j] = np.nan

                bad_rows_indices = []
                for j in range(len(small_df)):
                    ask_c = float(small_df['ask_c'].iloc[j])
                    ask_p = float(small_df['ask_p'].iloc[j])
                    bid_c = float(small_df['bid_c'].iloc[j])
                    bid_p = float(small_df['bid_p'].iloc[j])

                    a = not np.isnan(ask_c) and not np.isnan(ask_p)
                    b = not np.isnan(bid_c) and not np.isnan(bid_p)
                    ab = not np.isnan(ask_c) and not np.isnan(bid_p)

                    all_nan = np.isnan(ask_c) and np.isnan(ask_p) and \
                              np.isnan(bid_c) and np.isnan(bid_p)

                    if all_nan or not (a or b or ab):
                        bad_rows_indices.append(j)

                small_df.drop(bad_rows_indices, inplace=True)
                small_df = small_df.reset_index(drop=True)

                if len(small_df) > 0:
                    self.expiration_dates.append(e)
                    self.strikes.append(small_df['k'].astype(float).values)
                    for key in ['ask_c', 'ask_p', 'bid_c', 'bid_p']:
                        self.price[key].append(small_df[key].values)

        print(self.today, type(self.today))
        print(self.spot)
        print(self.expiration_dates, type(self.expiration_dates[0]))
        print(self.strikes)
        print(self.price)

    def get_data_from_file(self, call_put_diff, N1, N2, step=1):
        df_all = pd.read_csv(self.filename)

        # df_all.drop(df_all[df_all['Unnamed: 0'] == 'ENOENT: no such file or directory'].index, inplace=True)
        df_all.dropna(subset=['q'], inplace=True)

        today_list_without_seconds = [t[:-2] + '00' if t != 'q' else '' for t in df_all['q'].values]
        today_list_first_iteration, today_index_first_iteration = np.unique(today_list_without_seconds,
                                                                            return_index=True)

        today_unique = today_list_first_iteration[today_list_first_iteration != '']
        today_index = today_index_first_iteration[today_list_first_iteration != '']
        today_index[1:] -= 1

        today_unique = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in today_unique]

        zipped = sorted(zip(today_unique, today_index), key=lambda t: t[0])
        today_unique, today_index = zip(*zipped)

        expiry_real_unique = np.unique(df_all.e.values)
        expiry_real_unique = expiry_real_unique[expiry_real_unique != 'e']
        expiry_real_unique = sorted([datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in expiry_real_unique])

        max_len = len(expiry_real_unique)

        k_min = int(len(today_unique) % step) - 1

        n1 = len(today_unique) - 1
        n2 = len(today_unique)
        step_range = np.arange(n1, n2, step)

        # elif read_row == 'all':
        #     n1 = 0
        #     n2 = len(today_unique)
        #     step_range = []
        #     used_todays = []
        #     for i in range(len(today_unique) - 2):
        #         if today_unique[i].minute == 59 and today_unique[i + 1].minute != 0 \
        #                 or (today_unique[i].minute == 58 and today_unique[i + 1].minute != 59 and today_unique[
        #             i + 1].minute != 0 and today_unique[i + 2].minute != 0) \
        #                 or (today_unique[i].minute == 57 and today_unique[i + 1].minute != 58 and today_unique[
        #             i + 1].minute != 59 and today_unique[i + 1].minute != 0 and today_unique[i + 2].minute != 0 and
        #                     today_unique[i + 2].minute != 59) \
        #                 or today_unique[i].minute == 0:
        #             step_range.append(i)
        #             used_todays.append(today_unique[i])
        #             # print('Today', used_todays[-1])
        #         elif today_unique[i].minute == 1 and today_unique[i - 1].minute == 56:
        #             step_range.append(i)
        #             used_todays.append(today_unique[i])
        #             # print('Today', used_todays[-1])
        #         # else:
        #         # print('Datetime not found', today_unique[i])
        #
        # txt = pd.DataFrame({'today': today_unique})
        # txt.to_csv('today.txt')

        for k in step_range:
            today = today_unique[k]

            if k == len(today_unique) - 1:
                df_small = df_all.iloc[today_index[k]:].reset_index(drop=True)
            else:
                df_small = df_all.iloc[today_index[k]:today_index[k + 1] - 1].reset_index(drop=True)

            df_small = df_small.drop(df_small[df_small['k'] == 'k'].index).reset_index(drop=True)
            df_small = df_small.replace(0, np.nan)

            expiry_unique, expiry_index = np.unique(df_small.e.values, return_index=True)
            expiry_unique = expiry_unique[expiry_unique != 'e']
            expiry_unique = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in expiry_unique]

            zipped = sorted(zip(expiry_index, expiry_unique), key=lambda t: t[0])
            expiry_index_sorted, expiry_unique = zip(*zipped)

            expiry_unique_sorted = sorted(expiry_unique)

            df_list = []
            for i in range(1, len(expiry_unique)):
                df_list.append(
                    df_small.iloc[expiry_index_sorted[i - 1]:expiry_index_sorted[i]].reset_index(drop=True))
            df_list.append(df_small.iloc[expiry_index_sorted[-1]:].reset_index(drop=True))

            zipped = sorted(zip(expiry_unique, df_list), key=lambda t: t[0])
            _, df_list = zip(*zipped)
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

                    for j in range(len(df_list[i])):
                        strike = float(df_list[i]['k'][j])
                        spot = float(df_list[i]['s0'][j])

                        ask_c = float(df_list[i]['ask_c'][j])
                        ask_p = float(df_list[i]['ask_p'][j])
                        bid_c = float(df_list[i]['bid_c'][j])
                        bid_p = float(df_list[i]['bid_p'][j])

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
                    tmp_ask_c.append(df_list[i]['ask_c'].astype('float').values.tolist())
                    tmp_ask_p.append(df_list[i]['ask_p'].astype('float').values.tolist())
                    tmp_bid_c.append(df_list[i]['bid_c'].astype('float').values.tolist())
                    tmp_bid_p.append(df_list[i]['bid_p'].astype('float').values.tolist())
                else:
                    tmp_time_before_expiration.append(np.nan)
                    tmp_strikes.append([np.nan])
                    tmp_ask_c.append([np.nan])
                    tmp_ask_p.append([np.nan])
                    tmp_bid_c.append([np.nan])
                    tmp_bid_p.append([np.nan])

            self.today.append(today)
            self.spot.append(float(df_small.s0[0]))
            self.expiration_dates.append(expiry_unique_sorted)
            self.time_before_expiration.append(tmp_time_before_expiration)
            self.strikes.append(tmp_strikes)
            # print('!', tmp_strikes)
            self.price['ask_c'].append(tmp_ask_c)
            self.price['ask_p'].append(tmp_ask_p)
            self.price['bid_c'].append(tmp_bid_c)
            self.price['bid_p'].append(tmp_bid_p)
        # print(self.strikes)

        # self.cut_dates(N1, N2)


if __name__ == '__main__':
    file_path = 'C:/Users/admin/for python/HISTORY/options/BTC/new_version/OptBestBA_1m_Deribit_BTC_USDT_20210524_hourly.csv'
    # file_path = 'C:/Users/admin/for python/Futures_every_6h/FutBestBA_Deribit_BTC_USDT.csv'
    r = LastTableReader(file_path)
    r.get_data(0, 100, divided_by_spot=True)
