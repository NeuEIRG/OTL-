# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR                     
from sklearn.model_selection import train_test_split
from math import exp ,log ,sqrt,copysign
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
from sklearn.externals import joblib 
import random

class init_Dataset: # zhuoran9211
    
    # 数据初始化类，主要负责读取数据，划分数据集
    
    # 数据初始化函数，读取csv文件
    def __init__(self,filename):
        self.filename=filename
        self.df=pd.read_csv(filename)

    # 数据集划分函数
    def DataSetPartition(self,Test_size=0.3,Random_state=0):
        xnewdf = pd.DataFrame(self.df,columns= ['GrLivArea','TotRmsAbvGrd','FullBath','TotalBsmtSF','GarageCars','YearBuilt','OverallQual'])
        ynewdf = pd.DataFrame(self.df,columns= ['SalePrice'])
        ynewdf=ynewdf.values.reshape(-1,1) 
        X_train,X_test, y_train, y_test = train_test_split(xnewdf,ynewdf,test_size=Test_size, random_state=Random_state)
        return X_train,X_test, y_train, y_test


class Dataset:
    # 数据集类，保存训练集和验证集
    def __init__(self,X_train,X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    # 数据标准化/归一化函数
    def Normalized(self, methods):
        scale = eval(methods+"(feature_range=(0,1))")
        self.X_train=scale.fit_transform(X_train)
        self.X_test=scale.fit_transform(X_test)
        self.y_train=scale.fit_transform(y_train)
        self.y_test=scale.fit_transform(y_test)

class off_line_model:
    
    # 离线模型类，目前是svr
    def __init__(self,data,kernel):
        self.kernel = kernel
        self.data = data
        
    def Train_offline(self):
        clf = SVR(self.kernel).fit(self.data.X_train, self.data.y_train)
        joblib.dump(clf,'oldsvr.model')
    
    def ppredict(self,testx):
        old_model = joblib.load('oldsvr.model')
        result = old_model.predict(testx)
        return result

class KernelInfo:

    def __init__(self, tao, flag, x, type = "e"):
        self.tao = tao      # 该支持向量前面的系数
        self.flag = flag    # 该支持向量的标签值（对于分类问题来说就是真实的标签，而对于回归问题来说，这个y值是sign(y_real-y_pre)表示的是margin的正负
        self.x = x          # 支持向量
        self.type = type    # 所使用的核函数类型

class TOL:

    def __init__(self,dataset, off_line_model, ee, ploy_y, ploy_d, ploy_r, e_y, sig_y, tao, C, maxS):
        
        self.Dataset = dataset             # 加载数据集类
        self.OFFline_ML= off_line_model    # 加载离线模型类
        self.ee = ee                       # 超参数，ee是svr的管道宽度
        self.alpha11 = 0.9                 # 离线模型的系数
        self.alpha22 = 0.1                 # 在线PA算法的系数
        self.ploy_y = ploy_y    # 多项式核的超参数y
        self.ploy_d = ploy_d    # 多项式核的超参数d
        self.ploy_r = ploy_r    # 多项式核的超参数r
        self.e_y = e_y          # 径向基核的超参数e_y
        self.sig_y = sig_y      # sigmoid核超参数sig_y
        self.tao = tao          # 最新的（即t时刻）svr多项式的每一项前面的系数tao
        self.C = C              # 松弛变量前面的超参数C
        self.B = []             # B的每一元素都是KernelInfo类，里面保存了每一个时刻的支持向量的信息
        self.maxS = maxS        # 支持向量集合B
        self.S = 0              # 当前支持向量集合的大小
        
        self.set_old_wx = []
        self.set_new_wx = []
        self.set_old_yt = []
        self.set_real_yt = []
        
        self.cont = 0           #计数值，记录训练轮数的
        self.f = []             # the model-function of non-linear PA

    # 向量点乘函数
    def dot_mul(self,x1, x2):
        # 行向量乘以列向量结果是一个实数
        if len(x1) != len(x2):
            print("those Vectors latitudes are not same!!! in dot_mul")
        result = 0
        for i in range(len(x1)):
            result = result + x1[i] * x2[i]
        return result
    
    # 符号函数
    def sign(self,x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
        
    # 投影函数
    def st(self,xw, real_yt):
        temp = 0.5 * (xw-real_yt) * (xw-real_yt)
        return exp(temp)
    
    # 线性核函数
    def kernel_liner(self, x1, x2):
        return self.dot_mul(x1, x2)
    
    # 多项式核函数
    def kernel_polynomial(self, x1, x2):
        temp = self.ploy_y * self.dot_mul(x1, x2) + self.ploy_r
        return temp**self.ploy_d

    # 径向基核函数(高斯核函数)
    def kernel_e(self, x1, x2):
        temp=0
        for i in range(len(x1)):
            temp = (x1[i] - x2[i]) * (x1[i] - x2[i])+temp
        temp = -(self.e_y * temp)
        temp = exp(temp)
        return temp
    
    # sig核函数
    def kernel_sig(self, x1, x2):
        temp = self.sig_y * self.dot_mul(x1, x2)
        temp = (exp(temp) - exp(-temp))/(exp(temp) + exp(-temp))
        return temp
    
    # 参数tao计算公式（参数tao为PA算法主函数各项前的系数
    def cal_tao(self, loss, xt):
        summ=0
        for i in range(len(xt)):
            summ=xt[i]*xt[i]+summ
        temp = loss / summ 
        temp = min(self.C, temp)
        return temp
    
    # OTL单轮训练主函数
    def one_OTL(self, xt, real_yt):
        # 第几轮
        self.cont = self.cont+1
        # 根据旧的分类器预测出一个值，记为old_wx
        old_wx = self.OFFline_ML.ppredict([xt])
        new_wx = np.array([0.1])

        # 根据在线核PA算法预测出另一个值,记为new_wx
        for i in range(len(self.B)):
            new_wx = self.B[i].tao * self.B[i].flag * self.kernel_e(xt, self.B[i].x)+new_wx
            
        # 根据在线迁移学习算法HomOTL-I求出联合的预测结果yt
        yt = self.alpha11 * old_wx + self.alpha22 * new_wx   # 在线迁移学习的加权平均主函数
        flag = self.sign(real_yt-new_wx)                     # 核函数前的正负系数
        
        #print('旧svr模型预测值:',old_wx,'在线学习预测值:',new_wx,'在线迁移学习预测值:',yt,'真实值:',real_yt)
        #print( "alpha11的值为:",self.alpha11,"alpha22的值为:", self.alpha22)
        
        self.set_old_wx.append(old_wx)
        self.set_new_wx.append(new_wx)
        self.set_old_yt.append(yt)
        self.set_real_yt.append(real_yt)
        
        # 在线学习部分，求新旧预测结果的新权重
        temp1 = self.alpha11 * self.st(old_wx, real_yt)
        temp2 = self.alpha22 * self.st(new_wx, real_yt)
        new_alpha11 = temp1 / (temp1 + temp2)
        new_alpha22 = temp2 / (temp1 + temp2)
        
        # 在线学习部分：计算损失函数loss，更新tao值
        # loss = max(0, 1 - real_yt * new_wx) # 这个是分类问题的损失函数
        loss = max(0,abs(new_wx-real_yt)-self.ee)
        self.tao = self.cal_tao(loss, xt)

        # 在线学习部分，根据固定缓冲器的核在线学习算法更新在线学习的学习器
        if loss > 0: # prediction result is worry
            # 注意，对于非线性核来说，在B中增加支持向量就是相当于更新 w_t+1= w_t+tao*y*t
            new_support_vector = KernelInfo(self.tao, flag, xt) # 初始化新的支持向量（新建类KernelInfo）包含成员tao，real_yt,xt
            self.B.append(new_support_vector)      # 将新的支持向量加入到支持集合B中
            self.S = self.S +1                     # 将支持向量数量加1
            if self.maxS <= self.S:                # r如果支持向量的数量大于最大阈值，则需要随机剔除一个向量
                sel_x = self.B.pop(random.randint(0, len(self.B)-1))  # 在B的list中随机挑一个元素
                self.S = self.S - 1                # 
                self.f.append(new_support_vector)  # 加入新的支持向量
                self.f.remove(sel_x)               # 删除旧的支持向量
            else:
                self.f.append(new_support_vector) # 如果没满的话就直接加入到B中
            # end_if
        # end_tol
    
    # 在线训练模拟函数
    def train_TOL(self):
        # 模仿在线的形式进行在线迁移学习预测
        for i in range(len(self.Dataset.X_test)):
            bustol.one_OTL(self.Dataset.X_test[i],self.Dataset.y_test[i])
    
    # 绘图函数
    def plot_OTL(self):
        plt.figure(figsize=(20,7))
        plt.plot(list(range(len(self.set_old_wx))), self.set_old_wx, color='b',label="offline svr prediction")    # svr的预测值 set_old_yt
        plt.plot(list(range(len(self.set_new_wx))),self.set_new_wx , color='r',label="PA prediction")  #红线在线PA算法的预测值 set_real_yt
        plt.plot(list(range(len(self.set_real_yt))),self.set_real_yt , color='g',label="real value") # 绿线为真实值
        plt.plot(list(range(len(self.set_old_yt))),self.set_old_yt , color='y',label=" online transfer learning prediction") # 黄线为在线迁移学习的预测值
        plt.xlabel("the sequence of instance")
        plt.ylabel("values")
        plt.legend()
        plt.show()
    
    # 评价函数
    def Eva_OTL(self):
        print("OTL,MSE",mean_absolute_error(bustol.set_real_yt,bustol.set_old_yt))
        print("offline,MSE",mean_absolute_error(bustol.set_real_yt,bustol.set_old_wx))
        print("PA,MSE",mean_absolute_error(bustol.set_real_yt,bustol.set_new_wx))
        print("---------------------------------------------------------------------------")
        print("OTL,MAE",mean_absolute_error(bustol.set_real_yt,bustol.set_old_yt))
        print("offline,MAE",mean_absolute_error(bustol.set_real_yt,bustol.set_old_wx))
        print("PA,MAE",mean_absolute_error(bustol.set_real_yt,bustol.set_new_wx))
        print("---------------------------------------------------------------------------")
        print("OTL,EVS",explained_variance_score(bustol.set_real_yt,bustol.set_old_yt))
        print("offline,EVS",explained_variance_score(bustol.set_real_yt,bustol.set_old_wx))
        print("PA,EVS",explained_variance_score(bustol.set_real_yt,bustol.set_new_wx))
        print("---------------------------------------------------------------------------")
        print("OTL,r2_score",r2_score(bustol.set_real_yt,bustol.set_old_yt))
        print("offline,r2_score",r2_score(bustol.set_real_yt,bustol.set_old_wx))
        print("PA,r2_score",r2_score(bustol.set_real_yt,bustol.set_new_wx))

'''
        self.ee = ee    # 超参数，ee是svr的管道宽度
        self.a1pha11 = 0.5  # 迁移学习的老模型的参数，初始值为0.5
        self.alpha22 = 0.5  # 迁移学习的新模型的参数，初始值为0.5
        self.ploy_y = ploy_y    # 多项式核的超参数y
        self.ploy_d = ploy_d    # 多项式核的超参数d
        self.ploy_r = ploy_r    # 多项式核的超参数r
        self.e_y = e_y          # 径向基核的超参数e_y,从网上查的资料说libsvm里面e的值默认是1/k（其中k是类别数
        self.sig_y = sig_y      # sigmoid核超参数sig_y
        self.tao = tao          # 最新的（即t时刻）svr多项式的每一项前面的系数tao
        self.C = C              # 松弛变量前面的超参数C
        self.B = []             # B的每一元素都是KernelInfo类，里面保存了每一个时刻的支持向量的信息
        self.maxS = maxS        # 支持向量集合B
        self.S = 0              # 当前支持向量集合的大小
        
        我用的是径向基核函数，所以初始化其他核的参数就为0了
        参数设置说实话我一点谱也没有，完全是根据网上的经验给的
        超参数ee为0.1
        径向基核函数的超参数为0.1
        松弛变量前面的超参数C为0.5（我感觉有点太小了，但是如果这个数大了的话PA的结果更离谱）
        支持向量集合最大容量为13（太大不好）
'''

raw_data = init_Dataset("train.csv")
X_train,X_test, y_train, y_test=raw_data.DataSetPartition()
data = Dataset(X_train,X_test, y_train, y_test)
data.Normalized("MinMaxScaler")

svr = off_line_model(data,"rbf")
svr.Train_offline()

bustol = TOL(data,svr,0.09, 0,0,0, 0.8 ,0,0, 5 ,20)  # 初始化TOL实例，设置参数
bustol.train_TOL()
bustol.plot_OTL()
bustol.Eva_OTL()
