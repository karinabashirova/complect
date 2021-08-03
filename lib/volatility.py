from lib.useful_things import *
import lib.option_formulas as opt


class IVSurface:
    def __init__(self, times_before_expiration, expiration_dates, today, asset_prices, strike_prices, spot,
                 real_option_prices, price_type):
        self.times_before_expiration = times_before_expiration
        self.expiration_dates = expiration_dates
        self.today = today
        self.asset_prices = asset_prices
        self.spot = spot
        self.strike_prices = strike_prices
        self.real_option_prices = real_option_prices
        self.price_type = price_type
        self.options_type = OptionType.call if self.price_type[
                                                   -1] == 'c' else OptionType.put

        self.surface = self.get_IV_surface()

        self.delta = np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan)

        for n, time in enumerate(self.times_before_expiration):
            for m, strike in enumerate(self.strike_prices):
                self.delta[n][m] = opt.delta(asset_prices[n], strike, time, self.surface[n][m],
                                             self.options_type)

    def get_IV_surface(self):
        N = len(self.times_before_expiration)
        M = len(self.strike_prices)

        iv = np.zeros((N, M))

        all_points, nan_points = 0, 0
        for n in range(N):
            for m in range(M):
                try:
                    spot = self.asset_prices[n]
                except IndexError:
                    spot = self.spot
                strike = self.strike_prices[m]
                time = self.times_before_expiration[n]
                real_price = self.real_option_prices[n][m]

                if np.isnan(real_price):
                    iv[n][m] = np.nan
                else:
                    all_points += 1
                    iv[n][m] = self.count_iv_from_price(spot, strike, time, real_price)
                    if np.isnan(iv[n][m]):
                        nan_points += 1
        # pprint(f'all: {all_points}, nans: {nan_points}', 'y')
        return iv

    def count_iv_from_price(self, spot, strike, time, real_price, eps=0.05, max_iterations=100):
        if 0.8 * spot < strike < 1.2 * spot:
            implied_volatility = 1.1
        else:
            implied_volatility = 5.1

        # dv = eps + 1
        iterations = 0
        theoretical_price = 1e6

        while abs(theoretical_price - real_price) > eps and iterations <= max_iterations:
            theoretical_price = opt.price_by_BS(spot, strike, time, implied_volatility, self.options_type)
            vega = opt.vega(spot, strike, time, implied_volatility)

            dv = (theoretical_price - real_price) / vega
            # pprint(f'{vega}, {implied_volatility}, dv {dv}, diff {theoretical_price - real_price}', 'g')
            # dv = real_price / vega
            implied_volatility -= dv
            # if np.isnan(implied_volatility) or np.isnan(vega):
            #     pprint(f'\t{iterations}: theor_price {theoretical_price}, vega {vega}, dv {dv}, IV {implied_volatility}', 'y')

            iterations += 1
            if np.isnan(implied_volatility):
                break

        # if np.isnan(implied_volatility):
        #     print('RESULT IV', implied_volatility)
        #     pprint(
        #         f'VEGA,  spot {spot}, strike {strike}, time {time}, ' +
        #         f'real_price {real_price}, theor price {theoretical_price}, it {iterations}',
        #         'r')

        # print()
        # if implied_volatility > 5:
        #     implied_volatility = 5
        return implied_volatility if 0 < implied_volatility < 3 else np.nan


class SABRSurface:
    def __init__(self, times_before_expiration, expiration_dates, today, asset_prices, strike_prices, spot,
                 surface=None, delta=None, params=None):
        self.times_before_expiration = times_before_expiration
        self.expiration_dates = expiration_dates
        self.today = today
        self.asset_prices = asset_prices
        self.spot = spot
        self.strike_prices = strike_prices

        self.surface = surface
        self.delta = delta
        self.params = params

        self.interpolated_times = None
        self.interpolated_strikes = None
        self.interpolated_surface = None

    def evaluate(self, IV_obj_dict, IV_delta_dict, use_weights=False, cut=False):
        if use_weights:
            weights_dict = self.get_delta_weights(IV_delta_dict)
        else:
            weights_dict = None

        self.surface, self.params = self.get_SABR_surface(IV_obj_dict, IV_delta_dict, weights_dict, use_weights, cut)

        self.delta = {'surface_c': np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan),
                      'surface_p': np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan)}

        for n, time in enumerate(self.times_before_expiration):
            for m, strike in enumerate(self.strike_prices):
                self.delta['surface_c'][n][m] = opt.delta(self.asset_prices[n],
                                                          strike,
                                                          time,
                                                          self.surface[n][m],
                                                          OptionType.call)
                self.delta['surface_p'][n][m] = opt.delta(self.asset_prices[n],
                                                          strike,
                                                          time,
                                                          self.surface[n][m],
                                                          OptionType.put)

        self.interpolated_times = np.linspace(min(self.times_before_expiration), max(self.times_before_expiration), 100)
        self.interpolated_strikes = np.linspace(min(self.strike_prices), max(self.strike_prices), 100)
        self.interpolated_surface = np.zeros((100, 100))

        f = interpolate.interp2d(self.strike_prices, self.times_before_expiration, self.surface)
        for n, t in enumerate(self.interpolated_times):
            for m, k in enumerate(self.interpolated_strikes):
                self.interpolated_surface[n][m] = f(k, t)

    def get_delta_weights(self, IV_delta_dict):
        weights = {key: np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan) for key in keys}

        for n, time in enumerate(self.times_before_expiration):
            for m, strike in enumerate(self.strike_prices):
                d = np.array([abs(IV_delta_dict[key][n][m]) for key in keys])
                s = np.nansum(1 - d)

                for i, key in enumerate(keys):
                    weights[key][n][m] = (1 - d[i]) / s

        return weights

    def get_SABR_surface(self, IV_obj_dict, IV_delta_dict, weights_dict, use_weights, cut):
        N = len(self.times_before_expiration)
        M = len(self.strike_prices)

        sabr = np.zeros((N, M))
        params_dict = {'alpha': np.full(N, np.nan),
                       'beta': np.full(N, np.nan),
                       'nu': np.full(N, np.nan),
                       'rho': np.full(N, np.nan)}

        # bounds = ([0 + 1e-10, 0, 0, -1 + 1e-10],
        #           [np.inf, 1, np.inf, 1 - 1e-10])

        bounds = ([0 + 1e-5, np.inf], [0, 1], [0, np.inf], [-1 + 1e-5, 1 - 1e-10])

        start_params = np.array([0.1, 0.5, 5, 0.1])
        for n, time in enumerate(self.times_before_expiration):
            new_params = \
                minimize(self.objective_function, start_params, method='L-BFGS-B',
                         args=(time, n, IV_obj_dict, IV_delta_dict, weights_dict, use_weights, cut),
                         bounds=bounds)  # , max_nfev=2000
            # new_params = \
            #     least_squares(self.objective_function, start_params,
            #                   args=(time, n, IV_obj_dict, IV_delta_dict, weights_dict, use_weights, cut),
            #                   bounds=bounds, max_nfev=1000)
            # print(new_params.success, self.expiration_dates[n])
            new_params = new_params['x']  #

            params_dict['alpha'][n] = new_params[0]
            params_dict['beta'][n] = new_params[1]
            params_dict['nu'][n] = new_params[2]
            params_dict['rho'][n] = new_params[3]

            for m, strike in enumerate(self.strike_prices):
                sabr[n][m] = ql.sabrVolatility(strike, self.asset_prices[n], time, *new_params)
                # if np.isnan(sabr[n][m]):
                #     pprint(
                #         f'NaN SABR, {strike}, {self.asset_prices[n]}, {time}, {new_params}, {self.expiration_dates[n]}',
                #         'r')
                #     pprint(f'{type(strike)}, {type(self.asset_prices[n])}, {type(time)}', 'r')
                # else:
                #     pass
                # print(f'{type(strike)}, {type(self.asset_prices[n])}, {type(time)}')

                # print('GOOD SABR', strike, self.asset_prices[n], time, new_params, self.expiration_dates[n])

        if np.count_nonzero(np.isnan(sabr)) > 0 or \
                np.count_nonzero(np.isinf(sabr)) > 0 or \
                np.count_nonzero(sabr < 0) > 0:
            sabr = self.fill_nan(sabr)

        return sabr, params_dict

    def fill_nan(self, surface):
        for n, time in enumerate(self.times_before_expiration):
            if np.count_nonzero(np.isnan(surface[n])) > 0 or \
                    np.count_nonzero(np.isinf(surface[n])) > 0 or \
                    np.count_nonzero(surface[n] < 0) > 0:
                x = self.strike_prices[(~np.isnan(surface[n])) & (~np.isinf(surface[n])) & (surface[n] > 0)]
                y = surface[n][(~np.isnan(surface[n])) & (~np.isinf(surface[n])) & (surface[n] > 0)]

                f = PchipInterpolator(x, y)

                for m in np.argwhere((np.isnan(surface[n])) | (np.isinf(surface[n])) | (surface[n] < 0)):
                    surface[n][m] = f(self.strike_prices[m])
        return surface

    def objective_function(self, params, time, n, IV_obj_dict, delta_dict, weights_dict, use_weights, cut):
        diff = []

        for key in keys:
            for m, strike in enumerate(self.strike_prices):
                if cut:
                    delta_check = (IV_obj_dict[key].options_type == 'CALL' and 0.05 < delta_dict[key][n][m] < 0.95) or \
                                  (IV_obj_dict[key].options_type == 'PUT' and -0.95 < delta_dict[key][n][m] < -0.05)
                else:
                    delta_check = True

                is_good_case = not np.isnan(IV_obj_dict[key].surface[n][m]) and delta_check

                if is_good_case:
                    SABR_volatility = ql.sabrVolatility(strike, self.asset_prices[n], time, *params)

                    d = abs(SABR_volatility - IV_obj_dict[key].surface[n][m])

                    d = d * weights_dict[key][n][m] if use_weights else d

                    if not np.isinf(d) and not np.isnan(d):
                        diff.append(d)
                    # else:
                    #     print(d, SABR_volatility, IV_obj_dict[key].surface[n][m], weights_dict[key][n][m])
                    # diff.append(np.nan)

        return np.nansum(np.array(diff) ** 2)
        # return diff

    def interpolate_surface(self, time, strike):
        try:
            f = interpolate.interp2d(self.strike_prices, self.times_before_expiration, self.surface)
            if time < np.min(self.times_before_expiration):
                time = np.min(self.times_before_expiration)
            elif time > np.max(self.times_before_expiration):
                time = np.max(self.times_before_expiration)

            if strike < np.min(self.strike_prices):
                strike = np.min(self.strike_prices)
                print('min strike')
            elif strike > np.max(self.strike_prices):
                strike = np.max(self.strike_prices)
                print('max strike')
            return f(strike, time)[0]
        except:
            print('dfitpack.error', end=' ')
            print('len(times_before_expiration):', len(self.times_before_expiration), end=', ')
            print('len(strike_prices):', len(self.strike_prices))

            if min(self.strike_prices) <= strike <= max(self.strike_prices):
                f = interpolate.interp1d(self.strike_prices, self.surface[0])
                return f(strike)
            elif strike < min(self.strike_prices):
                return self.surface[0][0]
            elif strike > max(self.strike_prices):
                return self.surface[0][-1]

    def get_strike_by_new_delta(self, d):
        zipped = sorted(zip(self.delta['surface_c'], self.strike_prices), key=lambda t: t[0])
        delta_slice_call, strikes_call = np.array([a for a, b in zipped]), np.array([b for a, b in zipped])
        cs1 = PchipInterpolator(delta_slice_call, strikes_call)
        new_strike_call = cs1(d)

        zipped = sorted(zip(self.delta['surface_p'], self.strike_prices), key=lambda t: t[0])
        delta_slice_put, strikes_put = np.array([a for a, b in zipped]), np.array([b for a, b in zipped])
        cs1 = PchipInterpolator(delta_slice_put, strikes_put)
        new_strike_put = cs1(d)
        # print('d', d, 'strike', new_strike_call, new_strike_put)

        return new_strike_call, new_strike_put

    def surface_slice_for_new_time(self, time):
        f = interpolate.interp2d(self.strike_prices, self.times_before_expiration, self.surface)
        surface_slice = np.array([f(strike, time) for strike in self.strike_prices]).flatten()

        return surface_slice

    def get_strike_by_time_delta(self, t, d, option_type=OptionType.call):
        delta_slice = self.delta_slice_for_new_time(t, option_type)

        zipped = sorted(zip(delta_slice, self.strike_prices), key=lambda element: element[0])
        sorted_delta_slice, strikes = np.array([a for a, b in zipped]), np.array([b for a, b in zipped])

        sorted_delta_slice = np.array(sorted_delta_slice)
        strikes = np.array(strikes)

        sorted_delta_slice, indices = np.unique(sorted_delta_slice, return_index=True)
        strikes = strikes[indices]
        try:
            cs1 = PchipInterpolator(sorted_delta_slice, strikes)
            new_strike = cs1(d)
            if not np.min(delta_slice) <= d <= np.max(delta_slice):
                print(f'delta {d:.4f} not in [delta_min, delta_max] = [{np.min(delta_slice):.4f}, {np.max(delta_slice):.4f}] for time {t:.4f}')
                if d < np.min(delta_slice):
                    new_strike = np.max(self.strike_prices)
                else:
                    new_strike = np.min(self.strike_prices)
        except:
            print('surface:', self.surface)
            print('delta:', sorted_delta_slice, 'strikes:', strikes, 'd:', d, 't:', t)
            new_strike = 0
        # print('strike', new_strike, 'time', t, 'delta', d)
        return new_strike

    def get_vol_by_time_delta(self, t, d, option_type=OptionType.call):
        new_strike_call = self.get_strike_by_time_delta(t, d, option_type)

        return self.interpolate_surface(t, new_strike_call)

    def delta_slice_for_new_time(self, time, option_type=OptionType.call):
        if option_type == OptionType.call:
            try:
                f = interpolate.interp2d(self.strike_prices, self.times_before_expiration, self.delta['surface_c'])
                # print('delta slice', np.array([f(strike, time) for strike in self.strike_prices]).flatten())
                return np.array([f(strike, time) for strike in self.strike_prices]).flatten()
            except:
                print('dfitpack.error, call', time, self.strike_prices, self.times_before_expiration)
                print(self.delta['surface_c'])
                print()
                return self.delta['surface_c'][0]
        else:
            try:
                f = interpolate.interp2d(self.strike_prices, self.times_before_expiration, self.delta['surface_p'])
                return np.array([f(strike, time) for strike in self.strike_prices]).flatten()

            except:
                print('dfitpack.error, put', time, self.strike_prices, self.times_before_expiration)
                print(self.delta['surface_p'])
                print()
                return self.delta['surface_p'][0]


# def get_strike_by_delta(strikes, delta, d):
#     zipped = sorted(zip(delta, strikes), key=lambda t: t[0])
#     delta_slice_call, strikes_call = np.array([a for a, b in zipped]), np.array([b for a, b in zipped])
#     cs1 = PchipInterpolator(delta_slice_call, strikes_call)
#     new_strike_call = cs1(d)
#     return new_strike_call
#
#
# def delta_slice_for_new_time(times, strikes, delta, time):
#     f = interpolate.interp2d(strikes, times, delta)
#     new_delta = np.array([f(strike, time) for strike in strikes]).flatten()
#     return new_delta
#
#
# def interpolate_surface(volatility_surface, times, strikes, time, strike):
#     f = interpolate.interp2d(strikes, times, volatility_surface)
#     return f(strike, time)[0]
