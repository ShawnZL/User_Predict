import pandas as pd
import torch.nn as torch
import torch
import matplotlib.pyplot as plt
import numpy as np
# 数据情况
# UDmap是eid所标识的行为的属性，总共有9种属性，其中 key4 5 7 8 9 是离散值
# key1, 2, 3, 6是连续值
# x1-x8是用户属性 1,2,6,7,8是离散值 3 4 5是连续值或者长离散值
# eid是行为id，总共有43种
# 先看看每种eid对应的新旧用户有多少

# work 1 the relation between eid with target
def distribution():
    train_data = pd.read_csv('./data/train_new.csv')
    # 统计每小时下标签的分布
    hourly_label_counts = train_data.groupby('eid')['target'].value_counts().unstack().fillna(0)

    # 绘制每小时下标签分布的变化
    hourly_label_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Eid Label Distribution")
    plt.xlabel("Eid")
    plt.ylabel("Count")
    plt.legend(title='Label')
    plt.show()

def eid_1_0():
    data = pd.read_csv('./data/train.csv')
    eid = range(42)
    eid_dict = {}
    zero2one = []
    for id in eid:
        tempdata = data[data['eid'] == id]
        len_zero = len(tempdata[tempdata['target'] == 0])
        len_one = len(tempdata[tempdata['target'] == 1])
        eid_dict[id] = {'0': len_zero, '1': len_one}
        zero2one.append(len_one / (len_zero + len_one))
        if len_one == 0:
            print(id, " : ", len_zero)

    indexs = np.argsort(-np.array(zero2one))  # 排序序号
    indexs = [str(index) for index in indexs]
    zero2one = list(reversed(sorted(zero2one)))
    plt.bar(indexs, zero2one)
    plt.xlabel("Eid")
    plt.ylabel("The ratio between 0 and 1")
    plt.title("The ratio between 0 and 1 on every eid")
    plt.plot()
    plt.show()

# 得到下面这些eid，它们在训练集中没有新的用户，同时旧用户也很少，可以考虑剔除
# 同时，比例高低不一
ignore_eid = [6, 7, 16, 17, 18, 22, 23, 24]
ignore_eid_zero = [2, 11, 6, 1, 1, 1, 3, 1]

# end of work 1
# 以相同的方法观察用户侧特征 x1, x2, x6, x7, x8
# 以及行为侧key4, key 5, key7, key8, key9
def data_mining(you_want):
    data = pd.read_csv('./data/train_new.csv')
    features = list(set(data[you_want]))
    ratio = []
    for feature in features:
        tempdata = data[data[you_want]==feature]
        len_zero = len(tempdata[tempdata['target'] == 0])
        len_one = len(tempdata[tempdata['target'] == 1])
        ratio.append(len_one/(len_zero+len_one))
    indexs = [str(index) for index in np.argsort(-np.array(ratio))]
    ratio = list(reversed(sorted(ratio)))
    plt.bar(indexs, ratio)
    plt.xlabel("value of "+ you_want)
    plt.ylabel("The ratio of 1")
    plt.title("The ratio of 1 on every value of "+you_want)
    plt.plot()
    plt.show()

data_mining("x6")
# x6 的 2 3value 无用
# start of work 2



#
