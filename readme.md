## 购买预测模型

使用Session人脸和用户历史购买的离散特征，预测本Session会购买什么彩种

### 数据来源

* Session内抓取的第一张人脸的特征（百度人脸识别直接得到）： 
    
    美丑、表情、情绪，以及人脸id（可选，只针对已经存在人员）

* Session其他的特征：
     
    会话抓取人脸时间（0-24小时）

* 用Session人脸id匹配得到user id，再得到此用户历史订单的购买行为： 
    
    城市、常购买彩种、总购买天数、新老用户、常去站点类型（如超市、商场）  
        
    如果是新用户，不存在对应user id，历史购买特征用平均值或众数代替


### 数据处理

连续变量转为离散型one-hot，然后检验是否购买行为强相关，筛选后得到一共18个特征:
    
  
|  特征   | 数据来源  |
|  ----  | ----  |
| 美丑  | session人脸 |
| 大笑  | session人脸 |
| 无情绪  | session人脸 |
| 正面情绪  | session人脸 |
| 会话时间  | session |
| 宜昌  | 用户属性 |
| 恩施  | 用户属性 |
| 武汉  | 用户属性 |
| 好运十倍  | 历史购买 |
| 七乐彩  | 历史购买 |
| 十里桃花  | 历史购买 |
| 其他彩种  | 历史购买 |
| 总购买天数  | 历史购买 |
| 新老用户  | 历史购买 |
| 会所  | 历史购买 |
| 棋牌室  | 历史购买 |
| 超市  | 历史购买 |
| 餐饮  | 历史购买 |

### 模型

拼接特征后输入3层MLP做多分类，预测Session购买的彩种 (双色球，好运十倍，福彩3d，其他彩种)

### 回测结果

7月数据的准确率指标：

| 类别 | 准确率 |
| ---- | ---- |
| 平均 | 0.45 |
| 不买 | 0.533 |
| 好运十倍 | 0.214 |
| 双色球 | 0.261 |
| 福彩3d | 0.522 |
| 其他彩种 | 0.208 |


### 调用方法
```
python3 app.py \
    --port=8827 \
    --debug=False \
    --host='127.0.0.1' \
    --appname='buy_prediction' \
    --threaded=True
```


### 接口文档

http://wiki.szyh-smart.cn/dengqichun/ai/s4l0q