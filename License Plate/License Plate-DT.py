# -*- coding:utf-8 -*-

import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier



if __name__ =="__main__":
    #pandas读入数据
    path='license.csv'
    data=pd.read_csv(path)
    data['PriceAdded']=data['MinimumTranscationPrice']-data['WarningPrice']
    x=data[['Month', 'PersonalQuota', 'NumberOfAuctions', 'WarningPrice']]
    y=data['MinimumTranscationPrice']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    # reg=DecisionTreeRegressor(criterion='mse', max_depth=9)
    reg= DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    dt=reg.fit(x_train,y_train)

    y_hat=dt.predict(x_test)
    plt.plot(x_test, y_test, 'r*', ms=10, label='Actual')
    plt.plot(x_test, y_hat, 'go', linewidth=2, label='Predict')
    print y_test
    print y_hat
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
