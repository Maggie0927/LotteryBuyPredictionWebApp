import pymysql
import pandas as pd
import numpy as np
import datetime
from sshtunnel import SSHTunnelForwarder
import threading

class MysqlDB():

    def __init__(self):
        self.lottery_order = pd.read_csv('data/lottery_order.csv', header=0) #导入旧表
        self.lottery_order['支付时间'] = pd.to_datetime(self.lottery_order['支付时间'])
        self.db = ... # confidential information
        self.initialize()

    def get_db(self, date):
        '''
        提取数据库的彩票订单表
        '''


        sql = '''SELECT

        ds.STATION_TYPE AS stationType, o.OUT_USER_ID, um.CITY,
        o.PURCHASE_NUM, o.PAY_TIME, o.ORDER_AMOUNT,

        CASE
        WHEN olt.LOTTERYTYPE_NAME IS NOT NULL THEN olt.LOTTERYTYPE_NAME
        WHEN clt.LOTTERYTYPE_NAME IS NOT NULL THEN clt.LOTTERYTYPE_NAME
        END AS lotterytypeName

        from lottery_order o 
        LEFT JOIN device_site ds ON o.STATIONID = ds.STATION_ID
        LEFT JOIN user_member um ON o.USER_ID = um.ID
        LEFT JOIN ot_lottery_type olt ON o.LOTTERYTYPE_CODE = olt.LOTTERYTYPE_CODE
        LEFT JOIN ct_lottery_type clt ON o.LOTTERYTYPE_CODE = clt.LOTTERYTYPE_CODE

        where o.PAY_TIME >= '2021-03-01 00:00:00' and o.PAY_TIME <= '%s 23:59:59' ''' % (date)

        while True:
            try:
                self.db.ping()
                break
            except pymysql.err.OperationalError:
                self.db.ping(True)

        lottery_order = pd.read_sql(sql, self.db)
        # server.stop()
        # db.close()

        return lottery_order

    def convert_station_type(self, stationType):
        if pd.isnull(stationType):
            return '未知'

        map_type = {
            'LOTTERY_STORE': '彩票专卖店',
            'MARKET_STORE': '商场',
            'RESTAURANT_STORE': '餐饮',
            'SUPERMARKT_STORE': '超市',
            'KTV_STORE': 'KTV',
            'INTERNET_BAR_STORE': '网吧',
            'CHINEMA_STORE': '影院',
            'CHESS_CARD_ROOM_STORE': '棋牌室',
            'STATION_STORE': '车站',
            'CLUB_STORE': '会所',
            'BATHING_STORE': '洗浴',
            'BAR_STORE': '酒吧',
            'TEA_HOUSE_STORE': '茶社',
            'OTHER_STORE': '其他'
        }
        if stationType in map_type:
            return map_type[stationType]

        return '其他'

    def update_lottery_order(self):
        '''
        数据清洗
        '''
        date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        self.lottery_order = self.get_db(date)
        self.lottery_order.replace('2000-1-0 00:00:00', '',inplace=True)
        self.lottery_order['PAY_TIME'] = pd.to_datetime(self.lottery_order['PAY_TIME'])
        self.lottery_order.replace(pd.NaT, np.nan)
        self.lottery_order = self.lottery_order.dropna(subset = ['OUT_USER_ID'])
        # self.lottery_order = self.lottery_order[self.lottery_order['PAY_TIME'] != '2000-1-0 00:00:00']
        self.lottery_order['stationType'] = self.lottery_order.apply( \
        lambda x: self.convert_station_type(x.stationType), axis = 1)
        columns_map = {              
            'CITY': '城市', 
            'stationType': '站点类型',
            'lotterytypeName': '彩种名称',                     
            'OUT_USER_ID': '用户id',             
            'LOTTERYTYPE': '彩票类型',                              
            'PAY_TIME': '支付时间',                             
            'ORDER_AMOUNT': '订单金额',                              
        }
        self.lottery_order.rename(columns=columns_map, inplace = True)
        self.lottery_order['支付日期'] = self.lottery_order['支付时间'].map(lambda x: '%s-%s-%s' % (x.year,x.month,x.day))
        self.lottery_order.to_csv('data/lottery_order.csv', sep=',', index=False)
        print('update lottery_order at', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        timer = threading.Timer(86400, self.update_lottery_order)
        timer.start()

    def initialize(self):
        '''
        每天3：00定时更新
        '''
        now_time = datetime.datetime.now()
        next_time = now_time + datetime.timedelta(days=+1)
        next_year = next_time.date().year
        next_month = next_time.date().month
        next_day = next_time.date().day
        next_time = datetime.datetime.strptime(str(next_year)+"-"+str(next_month)+"-"+str(next_day)+" 03:00:00", "%Y-%m-%d %H:%M:%S")
        timer_start_time = (next_time - now_time).total_seconds()
        # timer_start_time = 0
        timer = threading.Timer(timer_start_time, self.update_lottery_order)
        timer.start()
