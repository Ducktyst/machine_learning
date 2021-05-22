from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np


#генерируем исходные данные: 750 строк-наблюдений и 14 столбцов-признаков
from sklearn.svm import SVR

np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))

#Задаем функцию-выход: регрессионную проблему Фридмана
Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 +
10*X[:,3] + 5*X[:,4]**5 + np.random.normal(0,1))

#Добавляем зависимость признаков
X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))

#линейная модель
lr = LinearRegression()
lr.fit(X, Y)


#гребневая модель
ridge = Ridge(alpha=7)
ridge.fit(X, Y)


#Лассо
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)

# Случайное Лассо
randomized_lasso = RandomizedLasso()
randomized_lasso.fit(X, Y)

# Рекурсивное сокращение признаков
estimator = SVR(kernel="linear")
rfe = RFE(estimator=estimator)
rfe.fit(X, Y)

# Линейная корреляция
f, pval = f_regression(X, Y, center=True)

# Регрессор на основе Случайного леса
rfr = RandomForestRegressor()
rfr.fit(X, Y)

print(lr.coef_)

names = ["x%s" % i for i in range(1,15)]

"""
Листинг 38 - 
"""

def rank_to_dict(ranks, names):
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14,1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)

    return dict(zip(names, ranks))

ranks = {}
ranks["Linear reg"] = rank_to_dict(lr.coef_, names)
ranks["Ridge"] = rank_to_dict(ridge.coef_, names)
ranks["Lasso"] = rank_to_dict(lasso.coef_, names)
ranks["RandomizedLasso"] = rank_to_dict(randomized_lasso.scores_, names)
ranks["RFE"] = rank_to_dict(rfe.ranking_, names)
ranks["RandomForestRegressor"] = rank_to_dict(rfr.feature_importances_, names)

# Создаем пустой список для данных
mean = {}

# Бежим по списку ranks
for key, value in ranks.items():
    # print("key=", key, "value = ",value)
    # Пробегаемся по списку значений ranks,
    # которые являются парой имя: оценка
    for item in value.items():
        # имя будет ключом для нашего mean
        # если элементов с текущим ключом в mean нет - добавляем
        if item[0] not in mean:
            mean[item[0]] = 0

        # суммируем значения по каждому ключу-имени признака
        mean[item[0]] += item[1]

# находим среднее по каждому признаку
for key, value in mean.items():
    res = value / len(ranks)
    mean[key] = round(res, 2)

# сортируем и распечатываем список
mean = sorted(mean.items(), key = lambda x: x[1], reverse=True)
# mean = sorted(mean.sort(ke))

print("MEAN")
print(mean, "\n")

print("Важные признаки: ")
for key, value in ranks.items():
    ranks[key] = sorted(value.items(), key = lambda item: item[1], reverse=True)

for key, value in ranks.items():
    print(key)
    print(value)


"""
Страница 211
"""