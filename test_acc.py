import json
import requests
import pymysql
import pandas as pd
import numpy as np
from sklearn import metrics
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import threading
from sshtunnel import SSHTunnelForwarder
from train import convert_lottery_type

api = 'http://127.0.0.1:8827/sir/api/buy-predict'

def get_data(date):
    '''
    获取每天的人脸和会话订单数据作为测试集输入
    '''

    server = ... # confidential information
    server.start()

    sql1 = '''
    select SESSION_ID, BEAUTY, EXPRESSION, EMOTION, FACE_ID_ONE, UNIX_TIMESTAMP(CREATE_TIME) as CREATE_TIME
    from user_session_face 
    where CREATE_TIME between '%s 00:00:00' and '%s 23:59:59' and DEVICE_SN = 'TYC201A00040'
    ''' % (date, date)

    sql2 = '''
    select SESSION_ID, LOTTERYTYPE_CODE
    from user_session_lottery_order 
    where CREATETIME between '%s 00:00:00' and '%s 23:59:59' and SESSION_ID is not NULL and DEVICE_SN = 'TYC201A00040'
    ''' % (date, date)

    while True:
        try:
            db = pymysql.connect(host = '127.0.0.1', \
                port = server.local_bind_port, user = 'platfromread', passwd = 'Szyc2019@', db = 'user_center', charset = 'utf8')
            db.ping()
            break
        except pymysql.err.OperationalError:
            db.ping(True)

    user_session_face = pd.read_sql(sql1, db)
    user_session_lottery_order = pd.read_sql(sql2, db)
    server.stop()
    db.close()

    data = pd.merge(user_session_face, user_session_lottery_order, how='left', on='SESSION_ID')
    if not data.empty:
        data['LOTTERYTYPE_NAME'] = data.apply(lambda x: convert_lottery_type(x.LOTTERYTYPE_CODE), axis = 1)
    return data

def test(date):
    '''
    测试并得到每天数据的auc
    '''
    data = get_data(date)
    if data.shape[0] == 0:
        return 0
    count = 0
    label = []
    time_start = time.time()
    for index, row in data.iterrows():
        try:
            req = requests.post(api,
                json={
                    'face_id':row.FACE_ID_ONE,
                    'beauty':row.BEAUTY,
                    'emotion':row.EMOTION,
                    'expression':row.EXPRESSION,
                    'create_time':row.CREATE_TIME,
                })
            res = req.json()
            if res['code'] != 200:
                continue
            res = req.json()['data']
            true = row.LOTTERYTYPE_NAME
            label.append(true)
            pred = label_map[max(res, key=res.get)]
            if true == pred:
                count += 1

        except Exception as e:
            return str(e)
    time_end = time.time()
    print('each request cost time:', round((time_end - time_start) * 1000 / data.shape[0],3), 'ms for date', date)
    # print(np.unique(label))
    return round(count / data.shape[0], 3)

def monitor_test():
    '''
    每天定时测并画图
    '''
    today = datetime.date.today().strftime('%Y-%m-%d')
    # today = '2021-08-07'
    acc = test(today)
    date.append(today)
    acc_list.append(acc)
    x = range(len(date))
    print(x, acc_list)
    plt.plot(x, acc_list, marker='o')
    plt.xticks(x, date)
    plt.xlabel('date')
    plt.ylabel('accuracy')
    plt.gcf().autofmt_xdate()
    plt.savefig('acc_result.jpg')
    # plt.show()
    timer = threading.Timer(86400, monitor_test)
    timer.start()

if __name__ == '__main__':
    date = []
    acc_list = []

    label_map = {
        "nobuy_prob": 0,
        "haoyunshibei_prob": 1,
        "shuangseqiu_prob": 2,
        "fucai3d_prob": 3,
        "other_prob": 4
    }

    now_time = datetime.datetime.now()
    now_year = now_time.date().year
    now_month = now_time.date().month
    now_day = now_time.date().day
    next_time = datetime.datetime.strptime(str(now_year)+"-"+str(now_month)+"-"+str(now_day)+" 23:45:00", "%Y-%m-%d %H:%M:%S")
    timer_start_time = (next_time - now_time).total_seconds()
    # timer_start_time = 0
    timer = threading.Timer(timer_start_time, monitor_test)
    timer.start()
