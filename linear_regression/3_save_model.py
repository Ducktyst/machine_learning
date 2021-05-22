import pandas
import pandas.core.series as Series
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplotasplt
from sklearn import linear_model
import pickle

#Определяем функцию сохранения
def SaveToObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

#Опредлеляем функцию загрузки
def LoadFromObject(filename):
    f = open(filename, 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    return loaded_obj

# Загружаем данные
data_train = pandas.read_csv('data-logistic.csv')
X = data_train.ix[:,1:3].values
y = (data_train.ix[:,0].values + 1)/2

#Для лучешго понимания представления данных распечатайте формы массивов
print (X.shape)
print (y.shape)

# Создание, обучение модели и получение ее прогноз
regression = linear_model.LogisticRegression()
regression.fit(X, y)
ans = regression.predict(X)
# Оценка качества созданной
metric = roc_auc_score(y, ans)
print("AUC Trained: ", metric.real)

#Сохранение и загрузка модели
SaveToObject(regression, "model")
loaded_model = LoadFromObject("model")

#Создание, обучение новой модели,
not_fitted = linear_model.LogisticRegression()
not_fitted.fit(X[0:2,:], y[0:2])
ans = not_fitted.predict(X)
metric = roc_auc_score(y, ans)
print("AUC not Trained: ", metric.real)
#получение прогноза и оценки качества загруженной модели

ans = loaded_model.predict(X)
metric = roc_auc_score(y, ans)
print ("AUCLoaded: ", metric.real)

