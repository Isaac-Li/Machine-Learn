# -*- coding:utf-8 -*-

import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ =="__main__":
    #pandas读入数据
    path='license.csv'
    data=pd.read_csv(path)
    data['PriceAdded']=data['MinimumTranscationPrice']-data['WarningPrice']
    x=data[['Month', 'PersonalQuota', 'NumberOfAuctions', 'WarningPrice']]
    y=data['MinimumTranscationPrice']
    # print x
    # print y

    #绘制图形
    mpl.rcParams['font.sans-serif']=[u'simHei']
    mpl.rcParams['axes.unicode_minus']=False

    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['Month'], y, 'ro')
    plt.title(u'月份')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['PersonalQuota'], y, 'ro')
    plt.title(u'车牌数量')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['NumberOfAuctions'], y, 'ro')
    plt.title(u'参拍人数')
    plt.grid()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['Month'], data['PriceAdded'], 'ro')
    plt.title(u'月份')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['PersonalQuota'], data['PriceAdded'], 'ro')
    plt.title(u'车牌数量')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['NumberOfAuctions'], data['PriceAdded'], 'ro')
    plt.title(u'参拍人数')
    plt.grid()

    plt.tight_layout()
    plt.show()

    #线性回归
    x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=3)
    # print x_train
    # print y_train
    linreg=LinearRegression()
    model=linreg.fit(x_train, y_train)

    print model
    print linreg.coef_
    print linreg.intercept_

    y_hat=linreg.predict(np.array(x_test))
    mse=np.average((y_hat-np.array(y_test))**2)
    rmse=np.sqrt(mse)
    print "mse & rmse:", mse, rmse
    print y_hat

    t=np.arange(len(x_test))
    plt.plot(t, y_test, 'ro-',linewidth=2,  label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    #Lasso mode
    # la_model=Lasso()
    la_model=Ridge()
    alpha_can=np.logspace(-5,1,100)
    lasso_model=GridSearchCV(la_model, param_grid={'alpha': alpha_can})
    lasso_model.fit(x_train, y_train)


    print '超参数：\f', lasso_model.best_params_

    y_lasso_hat=lasso_model.predict(np.array(x_test))
    print  '训练数据准确度：\f',lasso_model.score(x_train, y_train)
    print  '测试数据准确度：\f',lasso_model.score(x_test, y_test)
    mse_lasso=np.average((y_lasso_hat-np.array(y_test))**2)
    rmse_lasso=np.sqrt(mse_lasso)
    print mse_lasso, rmse_lasso
    print np.array(y_test)
    print y_lasso_hat

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_lasso_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'Lasso 线性回归预测价格', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    y_lasso_hat = lasso_model.predict(np.array(x))
    t = np.arange(len(x))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_lasso_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测价格', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # ----------------输入下月数据进行预测--------------------------------
    x_March=[[3,10000,200000,86000],[3,10000,210000,86000],[3,10000,220000,86000],[3,10000,230000,86000],[3,10000,240000,86000],[3,10000,250000,86000]]
    print x_March
    y_March_hat=lasso_model.predict(x_March)
    print y_March_hat

    t = np.arange(len(x_March))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    rects1= plt.bar(t, y_March_hat,width=0.35, facecolor = 'yellowgreen',edgecolor = 'white')
    plt.title(u'上海沪牌--3月份预测价格', fontsize=18)


    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height,
                    '%d' % int(height), ha='center', va='bottom')


    autolabel(rects1)

    plt.grid()
    plt.show()