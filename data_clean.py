import pandas as pd
import datetime
from mysqldb import MysqlDB
from collections import Counter
import heapq

face_to_user = pd.read_csv('data/face_to_user.csv', header=0)
face_user_dict = face_to_user.set_index('face_id')['user_id'].to_dict()
lottery_order = MysqlDB().lottery_order

def count_frequent_k(arr, k):
    '''
    得到arr中频率最高的前k个元素
    '''
    counts = dict(Counter(arr))
    if len(counts) <= k:
        return list(counts.keys())
    res = heapq.nlargest(k, counts.keys())
    ans = []
    for i in range(k):
        if counts[res[i]] >= len(arr) / k:
            ans.append(res[i])
        else:
            break
    return ans

def input_to_label(input_feature):
    '''
    将输入特征转为离散标签
    '''
    beauty = input_feature['beauty']
    expression = input_feature['expression']
    emotion = input_feature['emotion']
    session_time = datetime.datetime.fromtimestamp(input_feature['create_time'])
    session_hour = session_time.hour

    laugh = 1 if expression == 'laugh' else 0
    neutral_emotion = 1 if emotion == 'neutral' else 0
    positive_emotion = 1 if emotion == 'happy' else 0

    beauty_bins = [20, 30, 40, 50, 100]
    for i in range(len(beauty_bins)):
        if beauty <= beauty_bins[i]:
            beauty = i
            break

    session_hour_bins = [8, 12, 15, 17, 18, 21, 24]
    for i in range(len(session_hour_bins)):
        if session_hour <= session_hour_bins[i]:
            session_hour = i
            break

    if 'face_id' in input_feature:
        face_id = input_feature['face_id']
        if face_id in face_user_dict.keys():
            user_id = face_user_dict[face_id]
        else:
            user_id = ''
    else:
        user_id = ''

    # lottery_order = pd.read_csv('lottery_order.csv', header=0)
    # lottery_order['支付时间'] = pd.to_datetime(lottery_order['支付时间'])
    all_order = lottery_order[lottery_order['用户id'] == user_id]
    before_order = all_order[all_order['支付时间'] < session_time]
    if before_order.shape[0] == 0: #新用户，取全局平均值
        city = '未知'
        station_type = lottery_order['站点类型'].mode()[0]
        total_buy_day = 0
        qilecai = 0
        haoyunshibei = 0
        shilitaohua = 0
        other_lottery = 0
        old_user = 0 #新=0，老=1
    else:
        before_one_month_order = before_order[before_order['支付时间'] > (session_time - datetime.timedelta(days=30))]
        if before_one_month_order.shape[0] == 0: #过去一个月没有购买记录，取过去所有的购买记录
            axp = before_order
        else:
            axp = before_one_month_order
        city_list = axp['城市'].dropna()
        city = '未知' if len(city_list) == 0 else city_list.mode()[0]
        station_type = axp['站点类型'].mode()[0]
        total_buy_day = len(axp['支付日期'].unique())
        lottery_list = axp['彩种名称'].dropna().tolist()
        frequent_lottery = count_frequent_k(lottery_list, 3)
        qilecai = 1 if '七乐彩' in frequent_lottery else 0
        haoyunshibei = 1 if '好运十倍' in frequent_lottery else 0
        shilitaohua = 1 if '十里桃花' in frequent_lottery else 0
        other_lottery = 0
        for lottery in frequent_lottery:
            if lottery not in ['七乐彩','好运十倍','十里桃花']:
                other_lottery = 1
                break
        old_user = 1

    yichang = 1 if city == '宜昌' else 0
    enshi = 1 if city == '恩施' else 0
    wuhan = 1 if city == '武汉' else 0

    supermarket = 1 if station_type =='超市' else 0
    chess_room = 1 if station_type == '棋牌室' else 0
    club = 1 if station_type == '会所' else 0
    restaurant = 1 if station_type == '餐饮' else 0

    total_buy_day_bins = [1, 3, 5, 10, 20, 1000]
    for i in range(len(total_buy_day_bins)):
        if total_buy_day <= total_buy_day_bins[i]:
            total_buy_day = i
            break

    return [beauty, haoyunshibei, qilecai, shilitaohua, other_lottery, total_buy_day, old_user, laugh, neutral_emotion, \
        positive_emotion, session_hour, yichang, enshi, wuhan, club, chess_room, supermarket, restaurant]





