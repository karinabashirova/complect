import pandas as pd
import numpy as np
import datetime


class ReaderCutByDates:
    def __init__(self, filename):
        self.filename = filename
        self.expiration_dates = []
        self.time_before_expiration = []
        self.spot = []
        self.price = {'ask_c': [], 'ask_p': [], 'bid_c': [], 'bid_p': []}
        self.strikes = []
        self.today = []
        self.df = None

    def read_one_csv(self):
        df = pd.read_csv(self.filename, comment='k')
        # df.columns = ['req', 'k', 'bid_c', 'bid_size_c', 'ask_c', 'ask_size_c', 'mark_c', 'bid_p', 'bid_size_p',
        #               'ask_p', 'ask_size_p', 'mark_p', 'exp', 'q', 'u_price']
        # print(df.columns)
        df.columns = ['k', 'bid_c', 'ask_c', 'bid_p', 'ask_p', 'exp', 'q', 'u_price']
        # df = df[['k', 'bid_c', 'ask_c', 'bid_p', 'ask_p', 'exp', 'q', 'u_price']]
        # df.q = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df.q]
        # df.exp = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df.exp]
        # print(df)
        self.df = df.copy()
        # print(df)
        return df

    def cut_df(self, start_date, end_date):
        # print('start_date', start_date)
        # print('end_date', end_date)
        df_list = []
        df_to_cut = self.df.copy()
        prev_start_row = 0
        for i in range(1, len(df_to_cut.q)):
            if df_to_cut.q.iloc[i] != df_to_cut.q.iloc[i - 1]:
                # print(start_date, datetime.datetime.strptime(df_to_cut.q.iloc[prev_start_row],
                #                                            '%Y-%m-%d %H:%M:%S') , end_date)
                try:
                    if (start_date <= datetime.datetime.strptime(df_to_cut.q.iloc[prev_start_row],
                                                                 '%Y-%m-%d %H:%M:%S')) and (
                            datetime.datetime.strptime(df_to_cut.q.iloc[i - 1], '%Y-%m-%d %H:%M:%S') <= end_date):
                        # print(start_date, datetime.datetime.strptime(df_to_cut.q.iloc[prev_start_row], '%Y-%m-%d %H:%M:%S'),
                        #       datetime.datetime.strptime(df_to_cut.q.iloc[i - 1],
                        #                                  '%Y-%m-%d %H:%M:%S'), end_date)
                        if len(np.unique(df_to_cut.exp.iloc[prev_start_row:i])) > 1:
                            df_list.append(self.df.iloc[prev_start_row:i, :])
                except ValueError:
                    print(df_to_cut.q.iloc[prev_start_row])
                prev_start_row = i
                        # print(df_list)
        return df_list
