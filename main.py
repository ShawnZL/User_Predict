import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 导入模型
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# 导入交叉验证和评价指标
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import xgboost as xgb


def udmap_onehot(d):
    v = np.zeros(9) - 1
    # v = np.full(9, -1)
    if d == 'unknown':
        return v
    d = eval(d) # 将 'udmap' 的值解析为一个字典
    for i in range(1, 10): # 遍历 'key1' 到 'key9', 注意, 这里不包括10本身
        if 'key' + str(i) in d: # 如果当前键存在于字典中
            v[i-1] = d['key' + str(i)] # 将字典中的值存储在对应的索引位置上
    return v

def data_clean():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    """
    将udmap列进行one-hot编码
    #                    udmap  key1  key2  key3  key4  key5  key6  key7  key8  key9
    # 0           {'key1': 2}     2     0     0     0     0     0     0     0     0
    # 1           {'key2': 1}     0     1     0     0     0     0     0     0     0
    # 2  {'key1': 3, 'key2': 2}   3     2     0     0     0     0     0     0     0
    """
    # 使用 apply() 方法将 udmap_onethot 函数应用于每个样本的 'udmap' 列
    # np.vstack() 用于将结果堆叠成一个数组
    train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onehot)))
    test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onehot)))
    # 为新的特征 DataFrame 命名列名
    train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
    test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
    # 将编码后的 udmap 特征与原始数据进行拼接，沿着列方向拼接
    train_data = pd.concat([train_data, train_udmap_df], axis=1)
    test_data = pd.concat([test_data, test_udmap_df], axis=1)

    # 4. 编码 udmap 是否为空
    # 使用比较运算符将每个样本的 'udmap' 列与字符串 'unknown' 进行比较，返回一个布尔值的 Series
    # 使用 astype(int) 将布尔值转换为整数（0 或 1），以便进行后续的数值计算和分析
    train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
    test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)

    # 5. 提取 eid 的频次特征
    # 使用 map() 方法将每个样本的 eid 映射到训练数据中 eid 的频次计数
    # train_data['eid'].value_counts() 返回每个 eid 出现的频次计数
    train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
    test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())

    # 6. 提取 eid 的标签特征
    # 使用 groupby() 方法按照 eid 进行分组，然后计算每个 eid 分组的目标值均值
    # train_data.groupby('eid')['target'].mean() 返回每个 eid 分组的目标值均值
    train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
    test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())

    # 7. 提取时间戳
    # 使用 pd.to_datetime() 函数将时间戳列转换为 datetime 类型
    # 样例：1678932546000->2023-03-15 15:14:16
    # 注: 需要注意时间戳的长度, 如果是13位则unit 为 毫秒, 如果是10位则为 秒, 这是转时间戳时容易踩的坑
    # 具体实现代码：
    train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
    test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')

    # 使用 dt.hour 属性从 datetime 列中提取小时信息，并将提取的小时信息存储在新的列 'common_ts_hour'
    train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
    test_data['common_ts_hour'] = test_data['common_ts'].dt.hour

    train_data['common_ts_day'] = train_data['common_ts'].dt.day
    test_data['common_ts_day'] = test_data['common_ts'].dt.day

    train_data['x1_freq'] = train_data['x1'].map(train_data['x1'].value_counts())
    test_data['x1_freq'] = test_data['x1'].map(train_data['x1'].value_counts())
    train_data['x1_mean'] = train_data['x1'].map(train_data.groupby('x1')['target'].mean())
    test_data['x1_mean'] = test_data['x1'].map(train_data.groupby('x1')['target'].mean())

    train_data['x2_freq'] = train_data['x2'].map(train_data['x2'].value_counts())
    test_data['x2_freq'] = test_data['x2'].map(train_data['x2'].value_counts())
    train_data['x2_mean'] = train_data['x2'].map(train_data.groupby('x2')['target'].mean())
    test_data['x2_mean'] = test_data['x2'].map(train_data.groupby('x2')['target'].mean())

    train_data['x3_freq'] = train_data['x3'].map(train_data['x3'].value_counts())
    test_data['x3_freq'] = test_data['x3'].map(train_data['x3'].value_counts())

    train_data['x4_freq'] = train_data['x4'].map(train_data['x4'].value_counts())
    test_data['x4_freq'] = test_data['x4'].map(train_data['x4'].value_counts())

    train_data['x6_freq'] = train_data['x6'].map(train_data['x6'].value_counts())
    test_data['x6_freq'] = test_data['x6'].map(train_data['x6'].value_counts())
    train_data['x6_mean'] = train_data['x6'].map(train_data.groupby('x6')['target'].mean())
    test_data['x6_mean'] = test_data['x6'].map(train_data.groupby('x6')['target'].mean())

    train_data['x7_freq'] = train_data['x7'].map(train_data['x7'].value_counts())
    test_data['x7_freq'] = test_data['x7'].map(train_data['x7'].value_counts())
    train_data['x7_mean'] = train_data['x7'].map(train_data.groupby('x7')['target'].mean())
    test_data['x7_mean'] = test_data['x7'].map(train_data.groupby('x7')['target'].mean())

    train_data['x8_freq'] = train_data['x8'].map(train_data['x8'].value_counts())
    test_data['x8_freq'] = test_data['x8'].map(train_data['x8'].value_counts())
    train_data['x8_mean'] = train_data['x8'].map(train_data.groupby('x8')['target'].mean())
    test_data['x8_mean'] = test_data['x8'].map(train_data.groupby('x8')['target'].mean())

    # 保存csv
    train_data.to_csv("./data/train_new.csv", sep=",", index=None)
    test_data.to_csv("./data/test_new.csv", sep=",", index=None)

def find_x():
    # 判断x是否是数值类型
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')

    # 假设这是数据集中的字段名列表
    # field_names = ['uuid', 'eid', 'udmap', 'common_ts', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'target',
    #                'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9', 'udmap_isunknown',
    #                'eid_freq', 'eid_mean', 'common_ts_hour']
    field_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

    # 初始化一个字典用于存储字段类型信息
    field_types = {}

    # 遍历每个字段，分析其取值类型
    for field_index, field_name in enumerate(field_names):
        # 假设采用简单的规则：如果所有样本都可以转换为浮点数，则为数值类型；否则为类别类型
        is_numeric = all(isinstance(value, (int, float)) for value in test_data[field_name])
        field_types[field_name] = 'Numeric' if is_numeric else 'Categorical'

    # 打印字段类型信息
    for field_name, field_type in field_types.items():
        print(f"Field '{field_name}' is of type: {field_type}")

def print_viobox():
    # 绘制箱线图
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')
    field_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    flierprops = dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none')
    test_data[field_names].boxplot(flierprops=flierprops)
    # 设置标题和标签
    plt.title('Boxplot of x1 to x8 Fields')
    plt.ylabel('Values')
    plt.show()

def read_num():
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')

    field_names = ['x3', 'x7', 'x8']
    # 使用 describe() 函数统计数据基本情况
    field_stats = train_data[field_names].describe()

    # 打印统计结果
    print(field_stats)
    x3_not_41_count = train_data[train_data['x3'] != 41]['x3'].count()

    # 打印统计结果
    print("Number of records where x3 is not 41:", x3_not_41_count)

    x7_between_5_and_7_count = train_data[(train_data['x7'] >= 5) & (train_data['x7'] <= 7)]['x7'].count()

    # 打印统计结果
    print("Number of records where x7 is not 6:", x7_between_5_and_7_count)

    x8_not_0_count = train_data[train_data['x8'] != 1]['x8'].count()

    # 打印统计结果
    print("Number of records where x8 is not 1:", x8_not_0_count)


def print_heatmap():
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')

    # 相关性热力图
    sns.heatmap(train_data.corr().abs(), cmap='YlOrRd')

    # x7分组下标签均值
    sns.barplot(x='x7', y='target', data=train_data)

    plt.show()


def hour_change():
    train_data = pd.read_csv('./data/train_new.csv')
    # 统计每小时下标签的分布
    hourly_label_counts = train_data.groupby('common_ts_hour')['target'].value_counts().unstack().fillna(0)

    # 绘制每小时下标签分布的变化
    hourly_label_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Hourly Label Distribution")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Count")
    plt.legend(title='Label')
    plt.show()

def onthot():
    # 统计每个 key 对应的标签均值
    train_data = pd.read_csv('./data/train_new.csv')
    # 对 udmap 进行 One-Hot 编码
    udmap_encoded = pd.get_dummies(train_data['udmap'])

    # 将 One-Hot 编码后的 DataFrame 合并回原数据集
    train_data_encoded = pd.concat([train_data, udmap_encoded], axis=1)
    key_mean_values = {}
    for col in udmap_encoded.columns:
        key_mean_values[col] = train_data_encoded.groupby(col)['target'].mean()

    # 绘制直方图
    plt.figure(figsize=(12, 6))
    for col, mean_values in key_mean_values.items():
        plt.bar(mean_values.index, mean_values.values, label=col)

    plt.title("Mean Target Value by Key in udmap")
    plt.xlabel("Key")
    plt.ylabel("Mean Target Value")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def fotest():
    # data_clean()
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')
    """
    # 决策树算法
    
    clf = DecisionTreeClassifier()
    # 使用 fit 方法训练模型
    # train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1) 从训练数据集中移除列 'udmap', 'common_ts', 'uuid', 'target'
    # 这些列可能是特征或标签，取决于数据集的设置
    # train_data['target'] 是训练数据集中的标签列，它包含了每个样本的目标值
    clf.fit(
        train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
        train_data['target']  # 目标数据：将 'target' 列作为模型的目标进行训练
    )

    result_df = pd.DataFrame({
        'uuid': test_data['uuid'],  # 使用测试数据集中的 'uuid' 列作为 'uuid' 列的值
        'target': clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))
        # 使用模型 clf 对测试数据集进行预测，并将预测结果存储在 'target' 列中
    })
    """
    # 随机森林
    features = train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1)
    target = train_data['target']

    # 80作为训练，20作为验证
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 对训练数据进行标准化或者归一化处理
    X_val_scaled = scaler.transform(X_val) # 使用已经拟合好的scaler对象对X_val进行归一化处理
    # 这样可以确保在验证集上应用了与训练集相同的数据处理方式，从而保持一致性。

    # 构建随机森林模型
    model = RandomForestClassifier(n_estimators=91, max_features=10, oob_score=True, random_state=10)

    # 训练模型
    model.fit(X_train_scaled, Y_train)

    # 在验证集上进行预测
    predictions = model.predict(X_val_scaled)

    # 评估模型
    accuracy = np.mean(predictions == Y_val)
    print(f"Validation Accuracy: {accuracy:.7f}")

    #  确认最大特征数n_features
    # param_test1 = {"n_estimators": range(1, 101, 10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_test1,
    #                         scoring='roc_auc', cv=10)
    # gsearch1.fit(X_train_scaled, Y_train)
    #
    # print(gsearch1.best_params_)
    # print('best accuracy:%f' % gsearch1.best_score_)
    # 此时的结果为91

    # 大特征数max_features
    # param_test2 = {"max_features": list(range(1, 11, 1))}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=91),
    #                         param_grid=param_test2, scoring='roc_auc', cv=10)
    # gsearch1.fit(X_train_scaled, Y_train)
    #
    # print(gsearch1.best_params_)
    # print('best accuracy:%f' % gsearch1.best_score_)
    # 此时结果是10

    # 预测并保存结果
    X_test_scaled = scaler.transform(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))
    # 检查是否存在缺失值
    print(np.isnan(X_test_scaled).sum())

    # 使用 SimpleImputer 填充缺失值（如果有的话）
    imputer = SimpleImputer()
    X_test_scaled = imputer.fit_transform(X_test_scaled)
    test_predictions = model.predict(X_test_scaled)

    result_df = pd.DataFrame({
        'uuid': test_data['uuid'],
        'target': test_predictions
    })

    result_df.to_csv('submit.csv', index=None)

def K_transtest():
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')
    # 训练并验证SGDClassifier
    # pred = cross_val_predict(
    #     SGDClassifier(max_iter=10),
    #     train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
    #     train_data['target']
    # )
    # print(classification_report(train_data['target'], pred, digits=3))

    # 训练并验证DecisionTreeClassifier
    pred = cross_val_predict(
        DecisionTreeClassifier(),
        train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
        train_data['target']
    )
    print(classification_report(train_data['target'], pred, digits=3))

    # 训练并验证MultinomialNB 朴素贝叶斯
    # pred = cross_val_predict(
    #     MultinomialNB(),
    #     train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
    #     train_data['target']
    # )
    # print(classification_report(train_data['target'], pred, digits=3))

    # 训练并验证RandomForestClassifier
    pred = cross_val_predict(
        RandomForestClassifier(n_estimators=5),
        train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
        train_data['target']
    )
    print(classification_report(train_data['target'], pred, digits=3))

    # 验证xgboost
    pred = cross_val_predict(
        XGBClassifier(),
        train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
        train_data['target']
    )
    print(classification_report(train_data['target'], pred, digits=3))

def xg():
    # xgboost
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')
    # 随机森林
    features = train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1)
    target = train_data['target']

    # 80作为训练，20作为验证
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 对训练数据进行标准化或者归一化处理
    X_val_scaled = scaler.transform(X_val)  # 使用已经拟合好的scaler对象对X_val进行归一化处

    # 这样可以确保在验证集上应用了与训练集相同的数据处理方式，从而保持一致性。

    xgb_train = xgb.DMatrix(X_train_scaled, Y_train)
    xgb_test = xgb.DMatrix(X_val_scaled, Y_val)

    # 参数
    params = {'objective': 'multi:softmax', 'num_class': 3, 'booster': 'gbtree', 'max_depth': 5, 'eta': 0.1,
              'subsample': 0.7, 'colsample_bytree': 0.7}

    # 训练模型
    num_round = 50
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
    model = xgb.train(params, xgb_train, num_round, watchlist)

    # 在验证集上进行预测
    predictions = model.predict(xgb_test)

    # 评估模型
    accuracy = np.mean(predictions == Y_val)
    print(f"Validation Accuracy: {accuracy:.7f}")

    # 预测并保存结果
    X_test_scaled = scaler.transform(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))

    # 检查是否存在缺失值
    print(np.isnan(X_test_scaled).sum())

    # 使用 SimpleImputer 填充缺失值（如果有的话）
    imputer = SimpleImputer()
    X_test_scaled = imputer.fit_transform(X_test_scaled)
    xgb_test = xgb.DMatrix(X_test_scaled)

    test_predictions = model.predict(xgb_test)

    result_df = pd.DataFrame({
        'uuid': test_data['uuid'],
        'target': test_predictions
    })

    result_df.to_csv('submit2.csv', index=None)

def DicTree():
    train_data = pd.read_csv('./data/train_new.csv')
    test_data = pd.read_csv('./data/test_new.csv')

    # 决策树
    features = train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1)
    target = train_data['target']

    # 80作为训练，20作为验证
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 对训练数据进行标准化或者归一化处理
    X_val_scaled = scaler.transform(X_val)  # 使用已经拟合好的scaler对象对X_val进行归一化处理
    # 这样可以确保在验证集上应用了与训练集相同的数据处理方式，从而保持一致性。

    # 构建随机森林模型
    param = {'criterion': ['gini','entropy'], 'splitter': ['best', 'random'], 'max_depth': list(range(10,101,1)), 'min_samples_split': list(range(1, 10, 1)),'min_samples_leaf': [2, 3, 5, 10],
             'min_impurity_decrease': [0.1, 0.2, 0,3, 0.4, 0.5, 0.6]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=10)

    # 训练模型
    grid.fit(X_train_scaled, Y_train)

    # 在验证集上进行预测
    print('最优分类器:', grid.best_params_, '最优分数:', grid.best_score_)  # 得到最优的参数和分值
    # predictions = model.predict(X_val_scaled)
    #
    # # 评估模型
    # accuracy = np.mean(predictions == Y_val)
    # print(f"Validation Accuracy: {accuracy:.7f}")
    #
    # # 预测并保存结果
    # X_test_scaled = scaler.transform(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))
    # # 检查是否存在缺失值
    # print(np.isnan(X_test_scaled).sum())

    # 使用 SimpleImputer 填充缺失值（如果有的话）
    # imputer = SimpleImputer()
    # X_test_scaled = imputer.fit_transform(X_test_scaled)
    # test_predictions = model.predict(X_test_scaled)
    #
    # result_df = pd.DataFrame({
    #     'uuid': test_data['uuid'],
    #     'target': test_predictions
    # })
    #
    # result_df.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    data_clean()


