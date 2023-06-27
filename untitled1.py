from imblearn.over_sampling import SMOTE
from sklearn import metrics
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from WRELMfunction import RELM_train
from WRELMfunction import classifisor_test
from sklearn.model_selection import train_test_split
from KMeans import Cluster
import warnings
from KMeans_samples import KMeans
warnings.filterwarnings("ignore")

data=pd.read_excel(r'C:\Users\Dan\Desktop\DATAt+3(筛选后).xlsx')
cols=['AOC','NORC','COSM','RT','SROTS','MCR','SROE','NOP','CFOA/CL',\
      'ROE','OCIA','CE/DA','ART','ITR', 'FAT','TAT','OIGR','NAGR',\
          'TAGR','FAER','CR','LGR','TD/EBITDA','BPS','CFICR']
'类别变量'
cols1=['AOC','NORC','COSM','RT']
'数值变量'
cols2=['SROTS','MCR','SROE','NOP','CFOA/CL',\
      'ROE','OCIA','CE/DA','ART','ITR', 'FAT','TAT','OIGR','NAGR',\
          'TAGR','FAER','CR','LGR','TD/EBITDA','BPS','CFICR']
#'股权结构、公司治理、审计意见'中数值变量
cols3=['SROTS','MCR','SROE']
#其他类别中数值变量
cols4=['NOP','CFOA/CL','ROE','OCIA','CE/DA','ART','ITR', 'FAT','TAT','OIGR','NAGR',\
          'TAGR','FAER','CR','LGR','TD/EBITDA','BPS','CFICR']
X_data=pd.concat([data[cols2],data[cols1]],axis=1)
Y_data=data['y']
#分离标签与数据
num_train=0.2
'''
随机种子固定的话每次的训练集与测试集相同不会发生变化
'''
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=num_train,random_state=10)
'''
data_cate_train:训练集中分类型变量转换为独热编码
data_cate_test：测试集中分类型变量
data_cate_train_encoder：训练集中分类型变量的独热编码
data_cate_test_encoder：测试集中分类变量的独热编码
'''
data_cate_train=X_train[cols1]
data_cate_train_encoder=pd.get_dummies(data_cate_train)
data_cate_test = X_test[cols1]
data_cate_test_encoder=pd.get_dummies(data_cate_test)
'''
x_kmeans_train_all:包含全部指标的训练集
x_kmeans_test_all：包含全部指标的测试集
x_kmeans_test:测试集整合数值型和分类型变量
y_kmeans_train:训练集分类结果
y_kmeans_test:测试集分类结果
'''
x_kmeans_train_all=pd.concat([X_train[cols2],data_cate_train_encoder],axis=1)
#x_kmeans_train=pd.concat([X_train[cols3],data_cate_encoder],axis=1)
y_kmeans_train=Y_train
'''
用于聚类的指标：
SROTS:前十大股东持股比例
MCR：管理费用率
SROE：高管持股比例
AOC：审计意见类别,3类
NORC:公司属性，4类
COSM:上市板，3类
RT：关联交易，2类
'''
x_kmeans_test_all=pd.concat([X_test[cols2],data_cate_test_encoder],axis=1)
x_kmeans_test=x_kmeans_test_all.drop(cols4,axis=1) 
y_kmeans_test=Y_test
#查看各类数目
print('训练集样本中各类数目为：{}'.format(Counter(y_kmeans_train)))
print('测试集样本中各类数目为：{}'.format(Counter(y_kmeans_test)))
#对训练集SMOTE过采样
smo=SMOTE()
'''
x_smo_kmeans_train:过采样后用于K-Means++聚类的训练集特征数据
y_smo_kmeans_train:过采样后用于K-Means++聚类的训练集标签
'''
x_smo_kmeans_train_all,y_smo_kmeans_train=smo.fit_resample(x_kmeans_train_all, y_kmeans_train)
x_smo_kmeans_train=x_smo_kmeans_train_all.drop(cols4,axis=1)
min_max_scaler=MinMaxScaler()
#将标准化后的数据记为X_smo_kmeans_MinMax_train
X_smo_kmeans_MinMax_train=min_max_scaler.fit_transform(x_smo_kmeans_train)
#X_kmeans_test_MinMax：测试集归一化后数据
X_kmeans_test_MinMax=min_max_scaler.fit_transform(x_kmeans_test)
#最终测试集
data_smo_train=pd.concat([y_smo_kmeans_train,x_smo_kmeans_train_all],axis=1)
#打印结果
print("过采样后各类数目为:{}".format(Counter(y_smo_kmeans_train)))
A=np.array([1,1,1,2,2,2,5,5,5,5,1.5,1.5,1.5,1,1])
n_iter=100
all_index=[]
for k in range(2,3):
    #对不同的聚类个数得到类别标签clusters及簇质心centroids
    [clusters,centroids]=KMeans(X_smo_kmeans_MinMax_train,k,A,n_iter)  
    #计算轮廓系数
    b=metrics.silhouette_score(X_smo_kmeans_MinMax_train,clusters)
    print(k,'轮廓系数为',b)
    #根据训练集所得聚类中心，判断测试集中数据种类
    #clusters_test:测试集聚类结果
    clusters_test=Cluster(X_kmeans_test_MinMax,centroids,A)
    '''
    data_smo:将测试集聚类结果加入样本变量中
    '''
    data_smo_train['clusters']=clusters
    #data_smo_train.insert(loc=0, column='clusters', value=clusters)
    #将聚类标签加入测试集
    data_test=pd.concat([y_kmeans_test,x_kmeans_test_all],axis=1)
    data_test.insert(loc=0, column='clusters', value=clusters_test)
    '''
    K-Means将数据集分为k个类别，对每个类别样本数据用正则极限学习机做预测
    '''
    cols_dataframe=cols4+['y']+['clusters']+['results']