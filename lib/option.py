from lib.option_formulas import *
from lib.useful_things import *


def get_years_before_expiration(current_datetime, expiration_datetime):
    if type(current_datetime) is str:
        current_datetime = datetime.datetime.strptime(current_datetime, '%Y-%m-%d %H:%M:%S')

    if type(expiration_datetime) is str:
        expiration_datetime = datetime.datetime.strptime(expiration_datetime, '%Y-%m-%d %H:%M:%S')

    diff = expiration_datetime - current_datetime

    years = diff.total_seconds() / (365 * 24 * 60 * 60)

    if years < 0:
        raise ValueError(
            f'Current datetime {current_datetime} is later than expiration {expiration_datetime}')
    elif years == 0:
        raise ValueError(
            f'Current datetime {current_datetime} is equal to expiration {expiration_datetime}')
    else:
        return years


class Option:
    def __init__(self, current_datetime, expiration_datetime, current_price, strike_price, option_type,
                 underlying_asset='uASSET', founding_asset='fASSET'):
        self.__current_datetime = current_datetime
        self.__expiration_datetime = expiration_datetime
        self.__years_before_expiration = get_years_before_expiration(self.__current_datetime,
                                                                     self.__expiration_datetime)

        self.__current_price = current_price
        self.__strike_price = strike_price

        self.__option_type = option_type

        self.__underlying_asset = underlying_asset
        self.__founding_asset = founding_asset

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, value):
        self.__current_datetime = value
        self.__years_before_expiration = get_years_before_expiration(self.__current_datetime,
                                                                     self.__expiration_datetime)

    @property
    def time_before_expiration(self):
        return self.__years_before_expiration

    @property
    def option_type(self):
        return self.__option_type

    @property
    def strike_price(self):
        return self.__strike_price

    @property
    def asset_price(self):
        return self.__current_price

    @asset_price.setter
    def asset_price(self, value):
        self.__current_price = value

    @strike_price.setter
    def strike_price(self, value):
        self.__strike_price = value

    @property
    def expiration_datetime(self):
        return self.__expiration_datetime

    @expiration_datetime.setter
    def expiration_datetime(self, value):
        self.__expiration_datetime = value
        self.__years_before_expiration = get_years_before_expiration(self.__current_datetime, self.__years_before_expiration)

    @property
    def underlying_asset(self):
        return self.__underlying_asset

    @property
    def founding_asset(self):
        return self.__founding_asset

    def __str__(self):
        return f'{self.__underlying_asset}/{self.__founding_asset} --- ' + \
               f'{self.__option_type} {self.__expiration_datetime} {self.__strike_price:.2f} --- ' + \
               f'{self.__years_before_expiration:.4f} {self.__current_price:.2f}'

    def price(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return price_by_BS(S, K, T, volatility, self.__option_type)

    def delta(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return delta(S, K, T, volatility, self.__option_type)

    def gamma(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return gamma(S, K, T, volatility)

    def theta(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return theta(S, K, T, volatility)

    def vega(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return vega(S, K, T, volatility)

    def charm(self, volatility):
        S = self.__current_price
        K = self.__strike_price
        T = self.__years_before_expiration

        return charm(S, K, T, volatility)


if __name__ == '__main__':
    option = Option('2021-05-01 08:00:00', '2021-05-14 08:00:00', 1000, 1200, OptionType.put)
    print(option.price(0.9))
    option.current_datetime = '2021-05-09 08:00:00'
    option.expiration_datetime = '2021-05-19 08:00:00'
    print(option.price(0.9))
