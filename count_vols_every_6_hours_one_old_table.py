from apscheduler.schedulers.blocking import BlockingScheduler
from make_1h_vol_csv_last_table import make_vols_for_last_table
import sys
import datetime
import time
from Naked.toolshed.shell import execute_js, muterun_js

if __name__ == '__main__':
    start = time.time()
    print(start)
    filename = 'last_table_00.00.csv'

    make_vols_for_last_table(filename, 'BTC')
    print(time.time() - start)
