from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
from sklearn.externals import joblib
from keras import Sequential
from keras.layers import Dense
from scipy import integrate

def capcurve(y_values, y_preds_proba):
    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x': [0, rate_pos_obs, 1], 'y': [0, 1, 1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values, y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    #    y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False).reset_index('index', drop=True)
    y_cap_df_s = y_cap_df_s.sort_values(by=[1], ascending=False).reset_index(drop=True)

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)

    yy = np.append([0], yy[0:num_count - 1])  # add the first curve point (0,0) : for xx=0 we have yy=0

    percent = 0.5
    row_index = int(np.trunc(num_count * percent))
    #    print(row_index)
    #    print(type(yy))
    val_y1 = yy[row_index]
    val_y2 = yy[row_index + 1]
    if val_y1 == val_y2:
        val = val_y1 * 1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index + 1]
        val = val_y1 + ((val_x2 - percent) / (val_x2 - val_x1)) * (val_y2 - val_y1)

    sigma_ideal = 1 * xx[int(num_pos_obs) - 1] / 2 + (xx[num_count - 1] - xx[int(num_pos_obs)]) * 1
    sigma_model = integrate.simps(yy, xx)
    sigma_random = integrate.simps(xx, xx)

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    # ar_label = 'ar value = %s' % ar_value

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot(ideal['x'], ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx, yy, color='red', label='User Model')
    # ax.scatter(xx,yy, color='red')
    ax.plot(xx, xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1,
            label=str(val * 100) + '% of positive obs at ' + str(percent * 100) + '%')

    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("CAP Curve - a_r value =" + str(ar_value))
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    plt.show()
    return



# 导入数据集
file= 'C:/Users/yanka/Desktop/fix income/data.mat'
#csvfile = open('C:/Users/yanka/Desktop/fix income/trainset.csv')
#reader=csv.reader(csvfile)
data = sio.loadmat(file)
print(data)
X = data['data1'] #trainset1 39786*56矩阵
Y = data['data2'] #trainset2 39786*1矩阵
#%%
import pandas as pd
Xdf=pd.DataFrame(data=X)
Xdf.fillna(Xdf.mean(),inplace=True)
X=Xdf.values
#%%

print(X.shape)
print(Y.shape)

# 绘制原始数据图谱
plt.figure()
plt.plot(range(54), np.transpose(X))
#plt.plot(np.arange(1000, 50000, 2), np.transpose(X))
plt.xlabel('User loan info')
plt.ylabel('Credit Status')
plt.title('Credit default risk')
plt.show()

# 划分训练集、验证集与测试集
k = np.random.permutation(X.shape[0])
print(k)
X_train = X[k[:35808], :]      # 训练集
Y_train = Y[k[:35808], :]

X_test = X[35808:, :]       # 测试集
Y_test = Y[35808:, :]

# 归一化
mms = preprocessing.MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)       #利用训练集的标准作用于测试集，最后再外推。

Y_train = mms.fit_transform(Y_train)
Y_test = mms.transform(Y_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# 建立MLP模型
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(100,100),      #2层隐藏层,每层100个神经元
                  activation='tanh',
                  solver='sgd',
                  alpha=0.0001,
                  batch_size='auto',
                  learning_rate='adaptive',
                  learning_rate_init=0.1,
                  power_t=0.5,
                  max_iter=1000000,
                  shuffle=True,
                  random_state=None,
                  tol=0.0001,
                  verbose=False,
                  warm_start=False,
                  momentum=0.9,
                  nesterovs_momentum=True,
                  early_stopping=False,
                  validation_fraction=0.1,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  n_iter_no_change=10)



# 训练MLP模型
nn.fit(X_train, Y_train.ravel())

#joblib.dump(nn, 'nnet.model')
#
 #nn=joblib.load('nnet.model')
# MLP模型预测
y_sim = nn.predict(X_test)
Y_sim = mms.inverse_transform(y_sim.reshape(len(y_sim),-1))
# Y_sim = y_sim.reshape(10,-1)

Error = np.abs(Y_sim-Y_test) / Y_test
Result = np.hstack((Y_sim, Y_test, Error))#水平拼接
print(Result)

# 查看连接权值
 #print(nn.coefs_[0].shape)
#
 #print(nn.loss_)
 #print(nn.n_iter_)
# print(nn.n_layers_)


def compute_correlation(x,y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    ssr = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(0,len(x)):
        diff_xbar = x[i] - xbar
        dif_ybar = y[i] - ybar
        ssr += (diff_xbar * dif_ybar)
        var_x += diff_xbar**2
        var_y += dif_ybar**2
    sst = np.sqrt(var_x * var_y)
    return ssr/sst


R = compute_correlation(Y_sim, Y_test)
print(R**2)

#%%
def print_score(Y_sim, y_test):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    print("Test Result:\n")        
    print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, Y_sim)))
    #print("accuracy score: {0:.4f}\n".format(accuracy_score(Y_train, X_train)))
    print("Classification Report: \n {}\n".format(classification_report(y_test, Y_sim)))
    #print("Classification Report: \n {}\n".format(classification_report(Y_train, X_train)))
    print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, Y_sim)))
    #print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train, X_train)))


print_score(np.transpose(Y_sim).tolist()[0],np.transpose( Y_test).tolist()[0])
#print_score(np.transpose(X_train).tolist()[0],np.transpose( Y_train).tolist()[0])

#model=nn.fit(X_train, Y_train.ravel())
#from sklearn.externals import joblib
#joblib.dump(model,"fixincome.pkl")

classifier=nn
classifier.fit(X_train,Y_train)
# Creating CAP Curve
y_pred_proba = classifier.predict_proba(X=X_test)
Y_test=np.ravel(Y_test)
capcurve(Y_test, y_pred_proba[:,1])
