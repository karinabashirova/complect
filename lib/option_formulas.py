from lib.useful_things import *


def price_by_BS(S, K, T, sigma, option_type, r=0.0):
    if option_type == OptionType.call:
        return S * cdf(d1(S, K, T, sigma, r)) - K * np.exp(-r * T) * cdf(d2(S, K, T, sigma, r))
    elif option_type == OptionType.put:
        return K * np.exp(-r * T) * cdf(-d2(S, K, T, sigma, r)) - S * cdf(-d1(S, K, T, sigma, r))


def strike_from_delta(D, S, T, sigma, option_type, r=0.0):
    if option_type == OptionType.call:
        return S * np.exp(-ppf(D * np.exp(r * T)) * sigma * np.sqrt(T) + 0.5 * sigma ** 2 * T)
    elif option_type == OptionType.put:
        return S * np.exp(ppf(D * np.exp(r * T)) * sigma * np.sqrt(T) + 0.5 * sigma ** 2 * T)


def delta(S, K, T, sigma, option_type, r=0.0):
    if option_type == OptionType.call:
        return cdf(d1(S, K, T, sigma, r))
    elif option_type == OptionType.put:
        return cdf(d1(S, K, T, sigma, r)) - 1


def gamma(S, K, T, sigma, r=0.0):
    return pdf(d1(S, K, T, sigma, r)) / (S * sigma * np.sqrt(T))


def theta(S, K, T, sigma, option_type, r=0.0):
    value = -S * pdf(d1(S, K, T, sigma, r)) * sigma / (2 * np.sqrt(T))

    if option_type == OptionType.call:
        return value - r * K * np.exp(-r * T) * cdf(d2(S, K, T, sigma, r))
    elif option_type == OptionType.put:
        return value + r * K * np.exp(-r * T) * cdf(-d2(S, K, T, sigma, r))


def vega(S, K, T, sigma, r=0.0):
    return S * pdf(d1(S, K, T, sigma, r)) * np.sqrt(T) # / 100


def charm(S, K, T, sigma, r=0.0):
    # TODO check r
    return -pdf(d1(S, K, T, sigma, r)) * (-d2(S, K, T, sigma, r) / (2 * T))


def d1(S, K, T, sigma, r):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, sigma, r):
    return d1(S, K, T, sigma, r) - sigma * np.sqrt(T)


def cdf(value):
    return stats.norm.cdf(value)


def pdf(value):
    return stats.norm.pdf(value)


def ppf(value):
    return stats.norm.ppf(value)
