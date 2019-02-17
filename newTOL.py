# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

 #导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns       

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/train.csv")
xnewdf = pd.DataFrame(df,columns=['GrLivArea','TotRmsAbvGrd','FullBath','TotalBsmtSF','GarageCars','YearBuilt','OverallQual'])
ynewdf = pd.DataFrame(df,columns=['SalePrice'])
ynewdf=ynewdf.values.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler  # 归一化的库
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR                     # svr的库
from sklearn.model_selection import train_test_split

 X_train, X_test, y_train, y_test = train_test_split(xnewdf, ynewdf, test_size=.4, random_state=0)
 
 # 数据标准化
scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
y_train=scaler.fit_transform(y_train)
y_test=scaler.fit_transform(y_test)

# 训练离线svr
clf = SVR(kernel='rbf').fit(X_train, y_train)
clf.score(X_test, y_test)

# 保存模型
from sklearn.externals import joblib
joblib.dump(clf,'oldsvr.model')

from math import exp ,log ,sqrt,copysign
import random
# algorithm_34 CDOL
def pro_f(z):
    # this is the projection function
    temp = min(1, (z + 1) / 2)
    return max(0, temp)


def dot_mul(x1, x2):
    # 行向量乘以列向量结果是一个实数
    if len(x1) != len(x2):
        print("those Vectors latitudes are not same!!! in dot_mul")
    result = 0
    for i in range(len(x1)):
        result = result + x1[i] * x2[i]
    return result


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def st(xw, real_yt):
    temp = 0.5 * (pro_f(xw)-pro_f(real_yt)) * (pro_f(xw)-pro_f(real_yt))
    return exp(temp)
	
class KernelInfo:

    def __init__(self, tao, flag, x, type = "e"):
        self.tao = tao      # 该支持向量前面的系数
        self.flag = flag          # 该支持向量的标签值（对于分类问题来说就是真实的标签，而对于回归问题来说，这个y值是sign(y_real-y_pre)表示的是margin的正负
        self.x = x          # 支持向量
        self.type = type    # 所使用的核函数类型
		
class TOL:

    def __init__(self, ee, ploy_y, ploy_d, ploy_r, e_y, sig_y, tao, C, maxS):
        
        self.ee = ee    # 超参数，ee是svr的管道宽度
        self.alpha11 = 0.5
        self.alpha22 = 0.5
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
        
        self.cont = 0 #计数值，记录训练轮数的
        # -------------------------------------------------------------------

        self.f = []      # the model-function of non-linear PA

        #self.b = []      # the list of parameter of non-linear PA

    # end _int_

    def kernel_liner(self, x1, x2):
        return dot_mul(x1, x2)

    def kernel_polynomial(self, x1, x2):
        temp = self.ploy_y * dot_mul(x1, x2) + self.ploy_r
        return temp**self.ploy_d

    def kernel_e(self, x1, x2):
        temp=0
        for i in range(len(x1)):
            temp = (x1[i] - x2[i]) * (x1[i] - x2[i])+temp
        temp = -(self.e_y * temp)
        temp = exp(temp)
        return temp

    def kernel_sig(self, x1, x2):
        temp = self.sig_y * dot_mul(x1, x2)
        temp = (exp(temp) - exp(-temp))/(exp(temp) + exp(-temp))
        return temp

    def cal_tao(self, loss, xt):
        temp = loss / sqrt(dot_mul(xt, xt))       # type change?? should I force this type to float??
        temp = min(self.C, temp)
        return temp

    def new_loss(self, xt, real_yt):
        # 第几轮
        self.cont = self.cont+1
        # 根据旧的分类器预测出一个值，记为old_wx
        old_model = joblib.load('oldsvr.model')
        old_wx = old_model.predict([xt])
        new_wx = 0.5
        
        temp_cor =[]  # 由于在线核PA算法发生维度爆炸，为了防止起数据波动太大，这个是存放归一化之前的数据的容器
        temp_cor_tao = []

        # 每算一步都要对tao进行标准化
        
#         for i in range(len(self.B)):
#             temp_cor_tao.append(self.B[i].tao)
#         if len(temp_cor_tao)!=0:
#             temp_cor_tao=np.array(temp_cor_tao)
#             temp_cor_tao=scaler.fit_transform(temp_cor_tao.reshape(-1,1))
#         for i in range(len(self.B)):
#             self.B[i].tao = temp_cor_tao[i]
            
            
        # 根据在线核PA算法预测出另一个值,记为new_wx
        for i in range(len(self.B)):
            new_wx = self.B[i].tao * self.B[i].flag * self.kernel_e(xt, self.B[i].x)+new_wx
            
        # 每算一步都要进行标准化，将其结果映射到01区间中去，这样应该可以保证不会发生数值范围的巨大波动（应该可以吧。。但我觉得这不靠谱）
#         for i in range(len(self.B)):
#             temp_cor.append(self.B[i].tao * self.B[i].flag * self.kernel_e(xt, self.B[i].x))
            
#         if len(temp_cor)!=0:
#             temp_cor=np.array(temp_cor)
#             temp_cor=scaler.fit_transform(temp_cor.reshape(-1,1))
#             new_wx=sum(temp_cor)[0]

        # 根据在线迁移学习算法HomOTL-I求出联合的预测结果yt
        yt = self.alpha11 * pro_f(old_wx) + self.alpha22 * pro_f(new_wx)   # 这个是分类函数，如果是回归函数的应该还要修改
        
        flag = sign(real_yt-yt)
        
        print('旧svr模型预测值:',old_wx,'在线学习预测值:',new_wx,'在线迁移学习预测值:',yt,'真实值:',real_yt)
        
        
        self.set_old_wx.append(old_wx)
        self.set_new_wx.append(new_wx)
        self.set_old_yt.append(yt)
        self.set_real_yt.append(real_yt)
        
        # 在线学习部分，求新旧预测结果的新权重
        temp1 = self.alpha11 * st(old_wx, real_yt)
        temp2 = self.alpha22 * st(new_wx, real_yt)
        new_alpha11 = temp1 / (temp1 + temp2)
        new_alpha22 = temp2 / (temp1 + temp2)
        self.alpha11 = new_alpha11
        self.a1pha22 = new_alpha22

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
                print('')
                sel_x = self.B.pop(random.randint(0, len(self.B)-1))  # 在B的list中随机挑一个元素
                self.S = self.S - 1                # 
                self.f.append(new_support_vector)  # 加入新的支持向量
                self.f.remove(sel_x)    # it is OK ??? 删除旧的支持向量
            else:
                self.f.append(new_support_vector) # 如果没满的话就直接加入到B中
            # end_if
        # end_tol

    def tol_worry(self, xt, real_yt):
        pass
		
#   def _init_(self, ee,  ploy_y, ploy_d, ploy_r, e_y, sig_y, tao, C, maxS):

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


bustol = TOL(0.1, 0,0,0, 0.1,0,0, 0.5 ,13)  # 初始化TOL实例，设置参数

# 模仿在线的形式进行在线迁移学习预测
for i in range(len(X_test)):
    bustol.new_loss(X_test[i],y_test[i])

plt.figure(figsize=(20,7))
plt.plot(list(range(len(bustol.set_old_wx))), bustol.set_old_wx, color='b',label="offline svr prediction")    # svr的预测值 set_old_yt
plt.plot(list(range(len(bustol.set_new_wx))),bustol.set_new_wx , color='r',label="PA prediction")  #红线在线PA算法的预测值 set_real_yt
plt.plot(list(range(len(bustol.set_real_yt))),bustol.set_real_yt , color='g',label="real value") # 绿线为真实值
plt.plot(list(range(len(bustol.set_old_yt))),bustol.set_old_yt , color='y',label=" online transfer learning prediction") # 黄线为在线迁移学习的预测值
plt.xlabel("the sequence of instance")
plt.ylabel("values")
plt.legend()
plt.show()