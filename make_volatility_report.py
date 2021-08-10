import sys
from Naked.toolshed.shell import execute_js, muterun_js
import argparse

from IV_RV_regression import make_report

parser = argparse.ArgumentParser(description='Volatility surface')
parser.add_argument('n', type=str, help='File name for report')

args = parser.parse_args()

print('BTC options table reading ---', end=' ')

response = muterun_js(f'get_all_deribit_prices_mm_with_hats_v4.js 0 BTC >> last_table_BTC.csv')

if response.exitcode == 0:
    print('SUCCESS')
else:
    print('FAIL')
    sys.exit(1)

print('ETH options table reading ---', end=' ')

response = muterun_js(f'get_all_deribit_prices_mm_with_hats_v4.js 0 ETH >> last_table_ETH.csv')

if response.exitcode == 0:
    print('SUCCESS')
else:
    print('FAIL')
    sys.exit(1)

make_report(args.n)
