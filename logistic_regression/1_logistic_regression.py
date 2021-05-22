import numpy as np

# Подготовка данных
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)

Y = (X > 0).astype(np.float)

X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)
X = X[:, np.newaxis]


# Обучение модели
from sklearn import linear_model
log_reg = linear_model.LogisticRegression()
log_reg.fit(X, Y)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X, Y)


X_test = np.linspace(start=-5, stop=10, num=300)
X_test = X_test[:, np.newaxis]
y_log_label = log_reg.predict(X_test)
y_lin = lin_reg.predict(X_test)
y_log_probabilty = log_reg.predict_proba(X_test)[:,1]


# Отрисовка графика
import matplotlib.pyplot as plt

# рисуем исходные точки
plt.figure(1, figsize=(4, 3))
# plt.scatter(X.ravel(), Y, color='black', zorder=20)

# рисуем график меток логистической регрессии
# plt.plot(X_test, y_log_label, color='red', linewidth=3)

# рисуем график линейной регрессии
plt.plot(X_test, y_lin, linewidth=1)

# добавляем линию уровня, по которому точки будут относиться к 1 или 2 классу
plt.axhline(.5, color='green')

# рисуем логистическую кривую для вероятностей
# принадлежности объектов к классам
plt.plot(X_test, y_log_probabilty, color='blue', linewidth=3)

# настраиваем оси и выводим график
plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('LabelLogisticRegressionModel', 'LinearRegressionModel',
            'ProbabiltyLogRegression'), loc='lower right', fontsize='small')
plt.show()


# Подсчет качества логистической регрессии

y_test = (X_test > 0).astype(np.float)

from sklearn.metrics import accuracy_score

# оцениваем точность решений
lin_score = lin_reg.score(X_test, y_test)
print("Linear regression accuracy: ", lin_score)
log_score = accuracy_score(y_log_label, y_test)
print("Logistic regression accuracy: ", log_score)

"""
Листинг - 37
"""

