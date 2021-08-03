from lib.useful_things import *
import lib.option_formulas as opt


class ForwardPriceCounter:
    def __init__(self, spot, times_before_expiration, strike_prices, price):
        self.spot = spot

        self.times_before_expiration = times_before_expiration
        self.strike_prices = strike_prices

        self.price = price

        # self.ask_check = ask_check
        # self.bid_check = bid_check

    def count_average_forward(self, iv):
        forward = self.count_forward_from_parity()
        delta = self.count_delta(forward, iv)

        average_forward = np.full((len(keys), len(self.times_before_expiration)), np.nan)

        for k, key in enumerate(keys):
            for n in range(len(self.times_before_expiration)):
                delta_sum = np.nansum(delta[key][n])  # сумма по страйкам
                mean = np.nansum(forward[n] * delta[key][n] / delta_sum)

                average_forward[k][n] = mean if mean > 0 else np.nan

        average_forward_for_each_price_type = np.array(
            [self.forward_curve(average_forward[k]) * self.spot for k in range(len(keys))])

        average_forward_for_each_time = np.array(
            [np.nanmean(average_forward.T[n]) for n in range(len(self.times_before_expiration))])

        return np.nan_to_num(self.forward_curve(average_forward_for_each_time) * self.spot, nan=self.spot), \
               np.nan_to_num(average_forward_for_each_price_type, nan=self.spot)

    def count_forward_from_parity(self):
        forward = np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan)

        for n, time in enumerate(self.times_before_expiration):
            for m, strike in enumerate(self.strike_prices):
                # if self.ask_check[k]:
                ask_c = self.price['ask_c'][n][m]
                ask_p = self.price['ask_p'][n][m]
                bid_c = self.price['bid_c'][n][m]
                bid_p = self.price['bid_p'][n][m]

                if not np.isnan(ask_c) and not np.isnan(bid_p):
                    forward[n][m] = ask_c - bid_p + strike
                elif not np.isnan(ask_c) and not np.isnan(ask_p):
                    forward[n][m] = ask_c - ask_p + strike
                elif not np.isnan(bid_c) and not np.isnan(bid_p):
                    forward[n][m] = bid_c - bid_p + strike
                else:
                    warnings.warn('Not enough data for forward price counting, spot price instead')
                    forward[n][m] = self.spot

        return forward

    def count_delta(self, forward, iv):
        delta_dict = {key: np.full((len(self.times_before_expiration), len(self.strike_prices)), np.nan) for key in
                      keys}

        for n, time in enumerate(self.times_before_expiration):
            for m, strike in enumerate(self.strike_prices):
                # if self.ask_check[n]:
                for key in keys:
                    delta_dict[key][n][m] = opt.delta(forward[n][m],
                                                                  strike,
                                                                  time,
                                                                  iv[key].surface[n][m],
                                                                  iv[key].options_type)

        return delta_dict

    def forward_curve(self, forward):
        x = np.array(self.times_before_expiration)[~np.isnan(forward)]
        y = forward[~np.isnan(forward)] / self.spot - 1

        try:
            reg = LinearRegression().fit(x.reshape(-1, 1), y)
            yHat = reg.predict(x.reshape(-1, 1))

            # if yHat[0] > yHat[-1]:
            #
            #     my_pwlf = pwlf.PiecewiseLinFit(x, y)
            #
            #     def my_obj(x):
            #         t_my_obj = time.time()
            #
            #         l = y.mean() * 0.001
            #         f = np.zeros(x.shape[0])
            #
            #         for i, j in enumerate(x):
            #             my_pwlf.fit(j)
            #             f[i] = my_pwlf.ssr + (l * j)

            #         return f
            #
            #     # max_count = int(np.nanmin(self.time_to_expiration[k]) * 365)
            #
            #     max_count = 10
            #     bounds = [{'name': 'var_1', 'type': 'discrete',
            #                'domain': np.arange(2, max_count)}]
            #
            #     myBopt = BayesianOptimization(my_obj, domain=bounds, model_type='GP',
            #                                   initial_design_numdata=20,
            #                                   initial_design_type='latin',
            #                                   exact_feval=True, verbosity=True,
            #                                   verbosity_model=False)
            #     max_iter = 25
            #
            #     time_BayesianOptimization = time.time()
            #     myBopt.run_optimization(max_iter=max_iter, verbosity=False)
            #
            #     my_pwlf.fit(myBopt.x_opt)
            #
            #     x = x[::-1]
            #     yHat = []
            #     yHat = [my_pwlf.predict(x[0])[0]]
            #
            #     y_ = my_pwlf.predict(x)
            #     y_ = np.array(y_)
            #     for i in range(1, len(x)):
            #         # y_ = my_pwlf.predict(x[i])[0]
            #         # yHat.append(y_[i])
            #         if y_[i] <= yHat[i - 1]:
            #             yHat.append(y_[i])
            #         else:
            #             yHat.append(yHat[i - 1])
            #
            #     yHat = np.array(yHat[::-1])
            #     x = x[::-1]

            for n, f in enumerate(forward):
                if np.isnan(f):
                    x = np.insert(x, n, np.nan)
                    yHat = np.insert(yHat, n, np.nan)
        except ValueError:
            return y + 1

        return yHat + 1
