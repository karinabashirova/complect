from lib.useful_things import *
from lib.surface_creation import surface_object_from_file, surface_to_file
import lib.option_formulas as opt
from lib.option import get_years_before_expiration
import lib.plotter as pl


class HistoricalVolatility:
    def __init__(self, asset_file_name, rolling_size):
        self.file_name = asset_file_name
        self.spot_list = []
        self.historical_volatility_list = []
        self.rolling_size = rolling_size
        self.first_day = '2019-02-05 05-AM'

    def count_realised_volatility(self):
        self.spot_list = pd.read_csv(self.file_name, index_col='date')[['close']][::-1][self.first_day:]

        if self.rolling_size < len(self.spot_list):
            print('error')
            quit()

        asset = pd.read_csv(self.asset_file_name, index_col='date')[['close']][::-1][
                '2019-02-05 05-AM':last_day]

        self.historical_volatility_list = (np.log(asset / asset.shift(1)).rolling(24 * rolling_size).std()
                          * np.sqrt(24 * rolling_size)).close[24 * use_size:]
        self.asset_spot = asset.close[-1]