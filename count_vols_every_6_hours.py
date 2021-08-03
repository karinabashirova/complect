from apscheduler.schedulers.blocking import BlockingScheduler
from make_1h_vol_csv_last_table import make_vols_for_last_table
import sys
import datetime
import time
from Naked.toolshed.shell import execute_js, muterun_js

scheduler = BlockingScheduler({'apscheduler.timezone': 'UTC', 'misfire_grace_time': 60 * 60})

asset = 'BTC'
# asset = 'ETH'


@scheduler.scheduled_job('cron', hour='0,6,12,18')  # hour='1, 7, 13, 19'
def scheduled_job():
    print('ETH!!!!!!!!!!!')
    start = time.time()
    print(start)
    filename = f'last_table_{asset}.csv'

    print('RUN TIME: ', datetime.datetime.now())

    print('Options table reading ---', end=' ')
    response = muterun_js(f'get_all_deribit_prices_mm_with_hats_v4.js 0 {asset} >> {filename}')

    if response.exitcode == 0:
        print('SUCCESS')
    else:
        print('FAIL')
        sys.exit(1)

    make_vols_for_last_table(filename, asset)
    print(time.time() - start)


if __name__ == '__main__':
    print(asset)
    scheduler.configure()
    scheduler.start()
