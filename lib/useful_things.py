import warnings
import logging
import copy
import csv
import sys
import argparse
import datetime
from collections import namedtuple

import pandas as pd
import numpy as np

import QuantLib as ql

from scipy import stats
from scipy import interpolate
from scipy.optimize import minimize, Bounds, least_squares
from scipy.interpolate import PchipInterpolator

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

OptionType = namedtuple('OptionType', 'call, put')(call='CALL', put='PUT')

PriceType = namedtuple('PriceType', 'ask_c, bid_c, ask_p, bid_p')(
    ask_c='ask_c', bid_c='bid_c', ask_p='ask_p', bid_p='bid_p')

keys = ['ask_c', 'ask_p', 'bid_c', 'bid_p']


class Data:
    def __init__(self, strikes, times, dates, price, spot, today):
        self.strikes = strikes
        self.times = times
        self.dates = dates
        self.prices = price
        self.spot = spot
        self.today = today
        self.ask_check = [True] * len(times)
        self.bid_check = [True] * len(times)


def pprint(s, color=None):
    if color == 'r':
        print('\033[91m' + s + '\033[0m')
    elif color == 'g':
        print('\033[92m' + s + '\033[0m')
    elif color == 'y':
        print('\033[93m' + s + '\033[0m')
    elif color == 'b':
        print('\033[94m' + s + '\033[0m')
    elif color == 'm':
        print('\033[95m' + s + '\033[0m')
    elif color == 'c':
        print('\033[96m' + s + '\033[0m')
    elif color == 'w':
        print('\033[97m' + s + '\033[0m')
    else:
        print(s)