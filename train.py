import torch
import torch.nn as nn
from torch.nn.modules import activation
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import random
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
import datetime
import matplotlib.pyplot as plt
from sshtunnel import SSHTunnelForwarder
import pymysql

import warnings
warnings.filterwarnings('ignore')

time_step = "{0:%Y-%m-%d,%H-%M-%S/}".format(datetime.datetime.now())

from torch.utils.tensorboard import SummaryWriter 

class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 5)
        self.activation = nn.Tanh()
        self.activation = gelu()

    def forward(self, x, is_training=False):
        x = self.activation(self.linear1(x))
        if is_training:
            x = self.dropout(x)
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).to(device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))    
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).to(device)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)
        loss = batch_loss.mean()

        return loss

def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=2, epsilon=0.05):
    N = targets.size(0)
    # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    # 调用torch的log_softmax
    log_prob = F.log_softmax(outputs, dim=1)
    # 用之前得到的smoothed_labels来调整log_prob中每个值
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

def convert_lottery_type(x):
    lottery_map = {
        np.nan: 0, # 不买
        None: 0, # 不买
        'OTJSKJ0010': 1, # 好运十倍
        'SZYCOTDGG004': 1, # 好运十倍
        'SZYCCTSSQ001': 2, # 双色球
        'SZYCCTFCSD01': 3 # 福彩3d
    }
    if x in lottery_map:
        return lottery_map[x]
    return 4 # 其他彩种

def stratified_sample(data, label):
    #分层抽样
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    for i, j in ss.split(data, label):
        train_index, test_index = i, j
    train_data = data.loc[train_index, :]
    test_data = data.loc[test_index, :]
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

def get_data(date):
    '''
    从数据库提取date之前的数据表并合并
    '''
    # db = pymysql.connect(host = 'rm-wz9803mf0awg83z580o.mysql.rds.aliyuncs.com', \
    # port = 3306, user = 'platfromread', passwd = 'Szyc2019@', db = 'user_center', charset = 'utf8')

    server = SSHTunnelForwarder(
        ssh_address_or_host=('192.168.1.100', 22),
        ssh_username='mysql-aly',
        ssh_password='Szyh-2021',
        remote_bind_address=('rm-wz9803mf0awg83z580o.mysql.rds.aliyuncs.com', 3306))
    server.start()
    db = pymysql.connect(host = '127.0.0.1', \
    port = server.local_bind_port, user = 'platfromread', passwd = 'Szyc2019@', db = 'user_center', charset = 'utf8')

    sql = '''
    select SESSION_ID, BEAUTY, EXPRESSION, EMOTION, FACE_ID_ONE, UNIX_TIMESTAMP(CREATE_TIME) as CREATE_TIME
    from user_session_face 
    where CREATE_TIME between '2021-03-01 00:00:00' and '%s 23:59:59'            
    ''' % (date)

    user_session_face = pd.read_sql(sql, db)

    sql = '''
    select SESSION_ID, LOTTERYTYPE_CODE
    from user_session_lottery_order 
    where CREATETIME between '2021-03-01 00:00:00' and '%s 23:59:59' and SESSION_ID is not NULL
    ''' % (date)
    user_session_lottery_order = pd.read_sql(sql, db)
    server.stop()
    db.close()

    data = pd.merge(user_session_face, user_session_lottery_order, how='left', on='SESSION_ID')
    data['LOTTERYTYPE_NAME'] = data.apply(lambda x: convert_lottery_type(x.LOTTERYTYPE_CODE), axis = 1)
    return data

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'

    writer_train = SummaryWriter('summaries/train/' + time_step)
    writer_valid = SummaryWriter('summaries/valid/' + time_step)

    #设置随机种子
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    if device == 'cuda':
        torch.cuda.manual_seed(1)
    np.random.seed(1)

    dataset = pd.read_csv('data/dataset.csv', header=0) #加载训练数据
    x_list = ['美丑',\
        '好运十倍', '七乐彩', '十里桃花', '其他彩种', '总购买天数区间', \
            '新老用户', '大笑', '无情绪', '正面情绪', \
        '订单创建时间区间', '宜昌','恩施','武汉','会所','棋牌室','超市','餐饮']
    input_size = len(x_list)
    num_classes = 5

    # from data_clean import input_to_label
    # original_data = get_data('2021-07-31')
    # dataset = pd.DataFrame(columns=x_list + ['创建时间','购买彩种'], index=range(original_data.shape[0]))
    # for index, row in dataset.iterrows():
    #     if index % 1000 == 0:
    #         print(index)
    #     input_dict = {'face_id':original_data.loc[index, 'FACE_ID_ONE'],
    #                 'beauty':original_data.loc[index, 'BEAUTY'],
    #                 'emotion':original_data.loc[index, 'EMOTION'],
    #                 'expression':original_data.loc[index, 'EXPRESSION'],
    #                 'create_time':original_data.loc[index, 'CREATE_TIME'],
    #     }
    #     dataset.iloc[index, :] = input_to_label(input_dict) + [original_data.loc[index, 'CREATE_TIME'], original_data.loc[index, 'LOTTERYTYPE_NAME']]

    # dataset.to_csv('data/dataset.csv', sep=',', index=False)

    dataset['创建时间'] = dataset.apply(lambda x: datetime.datetime.fromtimestamp(x.创建时间), axis = 1)
    # train = dataset[dataset['创建时间'] <= datetime.datetime.strptime('2021-06-30 23:59:59', '%Y-%m-%d %H:%M:%S')]
    # test = dataset[dataset['创建时间'] >= datetime.datetime.strptime('2021-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')]

    train, test = stratified_sample(dataset, dataset['购买彩种'])
    test = dataset[dataset['创建时间'] >= datetime.datetime.strptime('2021-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')]

    train_features , train_labels = train[x_list].astype(float).values, train['购买彩种'].astype('int').values
    test_features, test_labels = test[x_list].astype(float).values, test['购买彩种'].astype('int').values

    #训练集上采样
    weights = 1 / np.unique(train['购买彩种'],return_counts=True)[1]
    # weights = np.array([0.1, 1.5, 1, 2.5, 3])
    sampler = WeightedRandomSampler(weights = weights[train_labels] * 10000, num_samples=train.shape[0], replacement=True)

    train_features = torch.tensor(train_features, dtype=torch.float).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    batch_size = 256
    num_epoch = 50
    init_lr = 1e-3
    # thres = 0.75
    alpha = 0.9
    gamma = 0.01

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    # train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # label_list = []
    # for i, (data, label) in enumerate(train_iter):
    #     label_list += label.numpy().tolist()
    # print(np.unique(label_list, return_counts=True))

    model = Net(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    loss_function = FocalLoss(alpha=alpha, gamma=gamma, class_num=num_classes).to(device)
    # loss_function = nn.CrossEntropyLoss()

    #学习率 warm up
    warm_up_epochs = 5
    warm_up_with_step_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs \
        else 0.5 * ( math.cos((epoch - warm_up_epochs) /(num_epoch - warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_step_lr)

    #训练和验证过程
    train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = [], [], [], []
    train_index, valid_index = 0, 0

    for epoch in range(1, num_epoch + 1):  
        train_loss, train_correct, train_total = 0, 0, 0
        valid_loss, valid_correct, valid_total = 0, 0, 0

        for i, (feature, label) in tqdm(enumerate(train_iter), total=train_features.shape[0]/batch_size+1):
            train_index += 1
            model.zero_grad()
            outputs = model(feature, is_training=True)
            loss = loss_function(outputs, label)
            # loss = CrossEntropyLoss_label_smooth(outputs, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

            train_loss += loss.item()
            # prob = F.softmax(outputs,dim=1)[:,1]
            # predicted = torch.tensor([1 if x >= thres else 0 for x in prob])
            predicted = torch.max(F.softmax(outputs,dim=1), 1)[1]
            train_total += len(label)
            train_correct += (predicted == label.detach().cpu()).sum().item()
            
            writer_train.add_scalar('Loss', loss.item(), train_index)
            writer_train.add_scalar('ACC', (predicted == label.detach().cpu()).sum().item() / len(label), train_index)
            writer_train.flush()

        train_loss_list.append(train_loss / (i + 1))
        train_accuracy_list.append(train_correct / train_total)
        
        for i, (feature, label) in enumerate(test_iter):
            valid_index += 1
            outputs = model(feature)
            loss = loss_function(outputs, label)
            # loss = CrossEntropyLoss_label_smooth(outputs, label)
            valid_loss += loss.item()
            # prob = F.softmax(outputs,dim=1)[:,1]
            # predicted = torch.tensor([1 if x >= thres else 0 for x in prob])
            predicted = torch.max(F.softmax(outputs,dim=1), 1)[1]
            valid_total += len(label)
            valid_correct += (predicted == label.detach().cpu()).sum().item()

            writer_valid.add_scalar('Loss', loss.item(), valid_index)
            writer_valid.add_scalar('ACC', (predicted == label.detach().cpu()).sum().item() / len(label), valid_index)
            writer_valid.flush()

        valid_loss_list.append(valid_loss / (i + 1))
        valid_accuracy_list.append(valid_correct / valid_total)

        scheduler.step()
        # writer_train.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        print('epoch: %d train loss: %.3f valid loss: %.3f train accuracy: %.3f valid accuracy: %.3f' %  \
            (epoch, train_loss_list[-1], valid_loss_list[-1], train_accuracy_list[-1], valid_accuracy_list[-1]))

    #验证集结果
    with torch.no_grad():
        outputs = model(test_features)
        # prob = F.softmax(outputs,dim=1)[:,1].detach().cpu().numpy()
        # predicted = torch.tensor([1 if x >= thres else 0 for x in prob])
        predicted = torch.max(F.softmax(outputs,dim=1), 1)[1]
        y_label = test_labels.detach().cpu().numpy()
        y_pred = predicted.detach().cpu().numpy()
        # fpr, tpr, thersholds = roc_curve(y_label, prob, pos_label=1)
        # roc_auc = auc(fpr, tpr)
        print(classification_report(y_label, y_pred, target_names=['0','1','2','3','4']))

        # 每个类别的准确率
        correct = [0] * num_classes
        total = [0] * num_classes
        for i in range(len(y_label)):
            label = y_label[i]
            correct[label] += (y_label[i] == y_pred[i])
            total[label] += 1
        for i in range(num_classes):
            print(f"Accuracy of class {i}: {round(correct[i] / total[i], 3)}")

    torch.save(model.state_dict(), 'config/model.pkl')