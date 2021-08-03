import pandas as pd
import numpy as np
import datetime

pd.options.mode.chained_assignment = None


class HistoricalReader:
    def __init__(self, filename):
        self.filename = filename
        self.expiration_dates = []
        self.time_before_expiration = []
        self.spot = []
        self.price = {'ask_c': [], 'ask_p': [], 'bid_c': [], 'bid_p': []}
        self.strikes = []
        self.today = []

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

    def get_data_from_file(self, call_put_diff, N1, N2, read_row='last', step=1):
        df_all = pd.read_csv(self.filename)  # , skiprows=19562072 + 705530 + 132339)

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
        expiry_real_unique = expiry_real_unique[expiry_real_unique != 'exp']
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

        elif read_row == 'all':
            n1 = 0
            n2 = len(today_unique)
            step_range = []
            used_todays = []
            for i in range(len(today_unique) - 2):
                if today_unique[i].minute == 59 and today_unique[i + 1].minute != 0 \
                        or (today_unique[i].minute == 58 and today_unique[i + 1].minute != 59 and today_unique[
                    i + 1].minute != 0 and today_unique[i + 2].minute != 0) \
                        or (today_unique[i].minute == 57 and today_unique[i + 1].minute != 58 and today_unique[
                    i + 1].minute != 59 and today_unique[i + 1].minute != 0 and today_unique[i + 2].minute != 0 and
                            today_unique[i + 2].minute != 59) \
                        or today_unique[i].minute == 0:
                    step_range.append(i)
                    used_todays.append(today_unique[i])
                    # print('Today', used_todays[-1])
                elif today_unique[i].minute == 1 and today_unique[i - 1].minute == 56:
                    step_range.append(i)
                    used_todays.append(today_unique[i])
                    # print('Today', used_todays[-1])
                # else:
                    # print('Datetime not found', today_unique[i])

        txt = pd.DataFrame({'today': today_unique})
        txt.to_csv('today.txt')

        for k in step_range:
            today = today_unique[k]

            if k == len(today_unique) - 1:
                df_small = df_all.iloc[today_index[k]:].reset_index(drop=True)
            else:
                df_small = df_all.iloc[today_index[k]:today_index[k + 1] - 1].reset_index(drop=True)

            df_small = df_small.drop(df_small[df_small['k'] == 'k'].index).reset_index(drop=True)

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

                        ask_c = float(df_list[i]['ask_c'][j])*spot
                        ask_p = float(df_list[i]['ask_p'][j])*spot
                        bid_c = float(df_list[i]['bid_c'][j])*spot
                        bid_p = float(df_list[i]['bid_p'][j])*spot

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
                        ask_c = float(df_list[i]['ask_c'][j])*spot
                        ask_p = float(df_list[i]['ask_p'][j])*spot
                        bid_c = float(df_list[i]['bid_c'][j])*spot
                        bid_p = float(df_list[i]['bid_p'][j])*spot

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
                    tmp_ask_c.append((spot*df_list[i]['ask_c'].astype('float').values).tolist())
                    tmp_ask_p.append((spot*df_list[i]['ask_p'].astype('float').values).tolist())
                    tmp_bid_c.append((spot*df_list[i]['bid_c'].astype('float').values).tolist())
                    tmp_bid_p.append((spot*df_list[i]['bid_p'].astype('float').values).tolist())
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
            self.price['ask_c'].append(tmp_ask_c)
            self.price['ask_p'].append(tmp_ask_p)
            self.price['bid_c'].append(tmp_bid_c)
            self.price['bid_p'].append(tmp_bid_p)

        self.cut_dates(N1, N2)


if __name__ == '__main__':
    file_name_btc = 'C:\\Users\\admin\\for python\\Futures\\deribit_eth_per_1min.csv'

    reader = HistoricalReader(file_name_btc)
    reader.get_data_from_file(0.1, 0, 100, 'all', 60)

    print([str(t) for t in reader.today[-25:]])

    prices = {'datetime': reader.today,
              'spot': np.full(len(reader.today), np.nan),
              'strike90': np.full(len(reader.today), np.nan),
              'strike100': np.full(len(reader.today), np.nan),
              'call90': np.full(len(reader.today), np.nan),
              'put90': np.full(len(reader.today), np.nan),
              'call100': np.full(len(reader.today), np.nan),
              'put100': np.full(len(reader.today), np.nan)
              }

    for k in range(len(reader.today)):
        if reader.spot[k] > 10000 and datetime.datetime(2021, 4, 23, 8, 0) in reader.expiration_dates[k]:
            index_expiration = \
                np.where(np.array(reader.expiration_dates[k]) == datetime.datetime(2021, 4, 23, 8, 0))[0][0]
            strikes = np.array(reader.strikes[k][index_expiration])

            strike90 = strikes[(0.9 < strikes / reader.spot[k]) & (strikes / reader.spot[k] < 1)]
            s90 = strike90[0] if len(strike90) > 0 else np.nan

            strike100 = strikes[(1 < strikes / reader.spot[k]) & (strikes / reader.spot[k] < 1.1)]
            s100 = strike100[-1] if len(strike100) > 0 else np.nan

            if s90 in reader.strikes[k][index_expiration] and s100 in reader.strikes[k][index_expiration]:
                index_strike90 = np.where(np.array(reader.strikes[k][index_expiration]) == s90)[0][0]
                index_strike100 = np.where(np.array(reader.strikes[k][index_expiration]) == s100)[0][0]
                # Покупаем
                # prices['call90'][k] = reader.price['ask_c'][k][index_expiration][index_strike90]  # real
                # prices['call90'][k] = reader.price['bid_c'][k][index_expiration][index_strike90]  # inverted
                prices['call90'][k] = (reader.price['bid_c'][k][index_expiration][index_strike90] +
                                       reader.price['ask_c'][k][index_expiration][index_strike90]) * 0.5  # mid
                # Продаем
                # prices['put90'][k] = reader.price['bid_p'][k][index_expiration][index_strike90]
                # prices['put90'][k] = reader.price['ask_p'][k][index_expiration][index_strike90]
                prices['put90'][k] = (reader.price['bid_p'][k][index_expiration][index_strike90] +
                                      reader.price['ask_p'][k][index_expiration][index_strike90]) * 0.5

                # Продаем
                # prices['call100'][k] = reader.price['bid_c'][k][index_expiration][index_strike100]
                # prices['call100'][k] = reader.price['ask_c'][k][index_expiration][index_strike100]
                prices['call100'][k] = (reader.price['bid_c'][k][index_expiration][index_strike100] +
                                        reader.price['ask_c'][k][index_expiration][index_strike100]) * 0.5

                # Покупаем
                # prices['put100'][k] = reader.price['ask_p'][k][index_expiration][index_strike100]
                # prices['put100'][k] = reader.price['bid_p'][k][index_expiration][index_strike100]
                prices['put100'][k] = (reader.price['bid_p'][k][index_expiration][index_strike100] +
                                       reader.price['ask_p'][k][index_expiration][index_strike100]) * 0.5

                prices['spot'][k] = reader.spot[k]
                prices['strike90'][k] = s90
                prices['strike100'][k] = s100

    pd.DataFrame(prices).to_csv('prices_different_strikes_mid.csv')
