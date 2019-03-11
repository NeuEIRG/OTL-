from math import ceil
from math import floor
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
 
# split a univariate dataset into train/test sets
# 训练集划分，将单变量数据集划分为池训练集和测试集
# 并且还有一点值得说明，训练LSTM的时候我们是以一天为step的训练的，而预测的时候是一周为step进行预测的，这也就是为什么要按周split数据集的原因
# 对于公交车数据来说，用30个历史数据预测当前到站时间
# 6698
def split_dataset(data):
    # 讲前三年的数据作为训练集，将最后一年的数据作为测试集
    train, test = data[0:4900], data[4900:6300]
    print(len(data))
    print(len(train))
    print(len(test))
    # 将训练数据重组为以周为单位的数据
    # split函数是一个numpy库的函数，其作用是把一个array从左到右按顺序切分，其
    # 切分长度不能超过array的元素个数,axis默认为０，即横向切分
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

def max_min(data):
    data=data.values
    Max=max(data)
    Min=min(data)
    temp=[0]*len(data)
    for i in range(len(data)):
        temp[i]=(data[i]-Min)/(Max-Min)
    return temp


def standardization(data):
    for i in range(len(data)):
        columns=data[i].columns
        for col in columns:
            data[i][col]=max_min(data[i][col])

def new_split_dataset(data):
    for i in range(len(data)):
        pass

   
# evaluate one or more weekly forecasts against expected values
# 根据预期值评价单周预测或者多周预测
def evaluate_forecasts(actual, predicted):
    # 参数说明：actual是实际值，predicted是预测值
    scores = list()
    # calculate an RMSE score for each day
    # 为所有周的每一天的预测值计算ＲＭＳＥ(均方根误差)评分
    for i in range(actual.shape[1]):
        # 计算平方误差
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # 计算均方根误差
        rmse = sqrt(mse)
        # 储存到scores容器中
        scores.append(rmse)
    # 计算均方根误差
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
 
# 计算得分的总和
def summarize_scores(name, score, scores):
    # join函数(python系统自带函数)是将列(list)表转化为字符串的函数，单引号中的逗号是分隔符。
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
   
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
    # 参数说明:n_input是滑动窗口大小,n_out是未来预测的步长，默认为7即说明我们要预测未来7天，即一周，的数据。
    # flatten data,数据扁平化
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    print('data shape',data.shape)
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input  # 预测的输入窗口截止索引(输入窗口大小：in_end-in_start=7)
        out_end = in_end + n_out     # 预测的输出窗口截止索引(输出窗口大小：out_end-in_end=7)
        # ensure we have enough data for this instance，保证输出窗口的移动不会超过数据集的边界
        if out_end < len(data):
#             x_input = data[in_start:in_end, 0]  # 由于是单特征预测，所以这里只取一个特征,x_input的结构是[1,2,...,8]这样的结构
#            # 下面rashape的目的是将输出数据x_input转化为2d的形式，即[[1],[2],[3],..,[8]]的形式，这个是为了满足keras模型的输入
#             x_input = x_input.reshape((len(x_input), 1))   # 这里要注意len()一个多维数组返回的是其最外层的维度大小
#             X.append(x_input)
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 12])  # 标签y无需转化为2D形式
        # move along one time step
        in_start += 1
    return array(X), array(y)
 
# train the model
def build_model(train, n_input):
    # prepare data,将时间序列数据转化为符合监督学习的格式
    train_x, train_y = to_supervised(train, n_input)
    print("in build_model")
    print("train_x, train_y",train_x.shape, train_y.shape)
    # define parameters，确定参数
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    print('n_timesteps, n_features, n_outputs',train_x.shape[1], train_x.shape[2], train_y.shape[1])
    print('input_shape',n_timesteps, n_features)
    # define model，定义模型结构
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network，拟合网络
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

'''
python不允许程序员选择采用传值还是传引用。
Python参数传递采用的肯定是“传对象引用”的方式。
这种方式相当于传值和传引用的一种综合。
如果函数收到的是一个可变对象（比如字典或者列表）的引用，
就能修改对象的原始值－－相当于通过“传引用”来传递对象。
如果函数收到的是一个不可变对象（比如数字、字符或者元组）的引用，
就不能直接修改原始对象－－相当于通过“传值'来传递对象。
'''

# make a forecast，进行一次预测
'''
forecast函数的预测规则：
n_input是滑动窗口的大小，即我们每次用最后n_input个周的历史数据去预测下一个周的数据，这个“历史数据”就来自
evaluate_model中history集合，即每次都用离待预测数据最近的n_input个连续数据去预测接下来最近时刻的情况，
这样充分利用了数据之间的时序信息，体现了时间序列模型与其他回归模型在实现上的不同。
'''
def forecast(model, history, n_input):
    # flatten data，数据扁平
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data，从输入数据中提取最近的观测值
    print('in forecast')
    print('data shape',len(data))
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, 1]，讲数据变换成符合lstm模型的输入格式
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week,预测下一周的数据
    print('length of input_x',input_x.shape)
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast，这个地方不太清楚，为什么只取第一项，应该和model.predict的返回值有关
    print('yhat',yhat)
    print('\n')
    yhat = yhat[0]
    return yhat
 
# evaluate a single model
# 使用的是前移评价(Walk Forward Validation)方法(时间序列模型中的k折交叉验证)
'''
evaluate_model中history数据集合的作用和更新规则:
在最初history是等于训练集train的，随后，在每一轮的预测中，每取出一个测试集的样例，在预测函数forcast调用结束
之后就将其加入到history集合中,最后history=train+test

'''
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    # 注意这里为什么不写成history=train,因为python中只有引用，没有赋值,所以必须将train"复制"一份才可以赋值给history
    history = [x for x in train]
    # walk-forward validation over each week，对每一次预测都进行前移评价
    predictions = list()
    for i in range(len(test)):
        # predict the week,得到一个test样例的预测结果
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions，储存预测结果
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week，讲该test样例当做历史数据加入到history数据集中作为下一次预测的输入
        history.append(test[i, :])
   
    # evaluate predictions days for each week，对每一周的预测结果进行评价
    predictions = array(predictions)
    mse = mean_squared_error(test[:, :, 12], predictions)
   
    score, scores = evaluate_forecasts(test[:, :, 12], predictions)
    return score, scores