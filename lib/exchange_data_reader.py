import sys

import pandas as pd
import numpy as np
import datetime
import logging


class ExchangeDataReader:
    def __init__(self, filename, max_day_count, min_day_count):
        self.filename = filename
        self.expiration_dates = []
        self.time_before_expiration = []
        self.price = {'ask_c': [], 'ask_p': [], 'bid_c': [], 'bid_p': []}
        self.strikes = []
        self.ask_check = []
        self.bid_check = []
        self.today = 0
        self.spot = 0
        self.max_day_count = max_day_count
        self.min_day_count = min_day_count

    def get_data_from_file(self):

        logging.basicConfig(filename='exchange_data_reader_error.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        try:
            num = self.get_row_to_start_read(self.filename, 'k')
            df = self.read_df(self.filename, num)
        except FileNotFoundError:
            logging.error(f'File {self.filename} does not exist')
            sys.exit(1)

        day_df = df['q']
        day_df.dropna(inplace=True)
        if (len(day_df)) == 0:
            logging.error("Empty column 'q' in file " + self.filename)
            sys.exit(1)

        self.today = day_df.values[0]

        expiry_real_unique, expiry_real_index = np.unique(df.e.values, return_index=True)
        zipped = sorted(zip(expiry_real_unique, expiry_real_index), key=lambda t: t[1])
        expiry_real_unique = [a for a, b in zipped]
        expiry_real_index = [b for a, b in zipped]

        df_list = []
        for i in range(1, len(expiry_real_unique)):
            df_list.append(df.iloc[range(expiry_real_index[i - 1], expiry_real_index[i]), :])
            df_list[i - 1] = df_list[i - 1].reset_index(drop=True)
        df_list.append(df.iloc[range(expiry_real_index[-1], len(df)), :].reset_index(drop=True))

        date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in expiry_real_unique]
        zipped = sorted(zip(date, expiry_real_unique, df_list), key=lambda t: t[0])
        expiry_real_unique = [b for a, b, c in zipped]
        df_list = [c for a, b, c in zipped]

        spot_df = df['s0']
        spot_df.dropna(inplace=True)
        if (len(spot_df)) == 0:
            logging.error("Empty column 's0' in file " + self.filename)
            sys.exit(1)

        self.spot = float(spot_df.values[0])

        today = pd.to_datetime(self.today, format='%Y-%m-%d %H:%M:%S')
        if self.max_day_count != -1:
            expiry_real_unique = np.delete(expiry_real_unique,
                                           np.where(pd.to_datetime(expiry_real_unique, format='%Y-%m-%d %H:%M:%S') - today >
                                                    datetime.timedelta(days=int(self.max_day_count), seconds=0, minutes=0,
                                                                       hours=0)))
        expiry_real_unique = np.delete(expiry_real_unique,
                                       np.where(pd.to_datetime(expiry_real_unique, format='%Y-%m-%d %H:%M:%S') - today <
                                                datetime.timedelta(days=int(self.min_day_count), seconds=0, minutes=0,
                                                                   hours=0)))
        for i, d in enumerate(expiry_real_unique):
            df_small = df_list[i]

            ok_list = set()

            list_ask = {'ask_c', 'ask_p'}
            list_bid = {'bid_c', 'bid_p'}

            if np.nansum(df_small['ask_c']) != 0:
                ok_list.add('ask_c')

            if np.nansum(df_small['ask_p']) != 0:
                ok_list.add('ask_p')

            if np.nansum(df_small['bid_c']) != 0:
                ok_list.add('bid_c')

            if np.nansum(df_small['bid_p']) != 0:
                ok_list.add('bid_p')

            self.ask_check.append(list_ask.issubset(ok_list))
            self.bid_check.append(list_bid.issubset(ok_list))

            self.error_writer(self.ask_check[-1], self.bid_check[-1], d)

            if self.ask_check[-1]:
                self.price['ask_c'].append(df_small['ask_c'].astype('float').values.tolist())
                self.price['ask_p'].append(df_small['ask_p'].astype('float').values.tolist())
            elif self.bid_check[-1]:
                self.price['ask_c'].append([])
                self.price['ask_p'].append([])

            if self.bid_check[-1]:
                self.price['bid_c'].append(df_small['bid_c'].astype('float').values.tolist())
                self.price['bid_p'].append(df_small['bid_p'].astype('float').values.tolist())
            elif self.ask_check[-1]:
                self.price['bid_c'].append([])
                self.price['bid_p'].append([])

            if self.ask_check[-1] or self.bid_check[-1]:
                expiration_date = pd.to_datetime(d, format='%Y-%m-%d %H:%M:%S')
                today = pd.to_datetime(self.today, format='%Y-%m-%d %H:%M:%S')

                diff = expiration_date - today

                try:
                    if diff < datetime.timedelta(days=0, seconds=0, minutes=0, hours=0):
                        logging.error("Date is incorrect")
                        raise ValueError
                except ValueError:
                    sys.exit(1)

                hours, remainder = divmod(diff.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                self.time_before_expiration.append(
                    diff.days / 365 + hours / (365 * 24) + minutes / (365 * 24 * 60) + seconds / (365 * 24 * 60 * 60))

                self.expiration_dates.append(expiry_real_unique[i])

                self.strikes.append(df_small['k'].astype('float').values.tolist())

            elif not self.ask_check[-1] and not self.bid_check[-1]:
                self.ask_check.pop()
                self.bid_check.pop()

    def read_df(self, file_name, num):
        try:
            df_all = pd.read_csv(file_name, skiprows=num)
        except FileNotFoundError:
            logging.error("File " + file_name + " does not exists")
            sys.exit(1)

        except pd.errors.EmptyDataError:
            logging.error("File " + file_name + " is empty")
            sys.exit(1)

        for c in ['k', 'e']:
            if df_all[str(c)].isnull().values.any():
                logging.error("Not enough values in file " + file_name + " in column '"
                              + str(c) + "'")
                sys.exit(1)

        if list(df_all.columns) != ['Unnamed: 0', 'k', 'bid_c', 'ask_c', 'mark_c', 'bid_p', 'ask_p',
                                    'mark_p', 'e', 'q', 'qty_p', 'qty_c', 's0']:
            logging.error("Not enough columns in file " + file_name)
            sys.exit(1)

        return df_all

    def get_row_to_start_read(self, filename, cell_text):
        num = 0
        with open(filename) as file:
            for i, row in enumerate(file):
                tmp = row.strip().split(',')
                if tmp[1] == cell_text:
                    num = i
        return num

    def error_writer(self, ask_check, bid_check, d):
        if ask_check and not bid_check:
            logging.warning("Empty column for bid for time " + d + ", only ask is used")
        elif bid_check and not ask_check:
            logging.warning("Empty column for ask for time " + d + ", only bid is used")
        elif not ask_check and not bid_check:
            logging.warning("Empty columns for ask and bid for time " + d + ", expiration date was dropped")
