import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# функция расчета медианы
def median(lst):
    return np.median(np.array(lst))


# Загрузка тренировочных данных
data_train = pandas.read_csv('perceptron-train.csv')
X_train = data_train.iloc[:, 1:3].values
y_train = data_train.iloc[:, 0].values

# Загрузка тестовых данных
data_test = pandas.read_csv('perceptron-test.csv')
X_test = data_test.iloc[:, 1:3].values
y_test = data_test.iloc[:, 0].values

# Инициализация массива для счетчика итераций
rs = np.linspace(0, 100, num=100)

# Инициализация списков для сохранения accuracy моделей
acc_p = []
acc_pn = []
acc_mlp = []
acc_mlpn = []

acc_mlpn_lbfgs = []
acc_mlpn_logistic = []

# настройки персептрона
max_iter = 2000
tol = 0.00000001

# Цикл прогона моделей
for i in rs:
    i = int(i)

    # Распечатка номера итерации
    print("Random: ", i)

    # Создание модели персептрона
    clf = Perceptron(random_state=i, alpha=0.01, max_iter=max_iter)

    # Обучение модели
    clf.fit(X_train, y_train)

    # Получение прогноза
    predictions = clf.predict(X_test)

    # Расчет показателя accuracy
    acc = accuracy_score(y_test, predictions)

    # Распечатка результата
    print("Perceptron: ", acc)

    # Добавление оценки в список оценок для модели персептрона
    acc_p.append(acc)

    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Работа пресептрона с нормализованными данными
    clf = Perceptron(random_state=i, alpha=0.01, max_iter=max_iter)
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print("Perceptron with normalization: ", acc)
    acc_pn.append(acc)

    # Создание многослойного классификатора
    mlp = MLPClassifier(random_state=i,
                        solver="sgd",
                        activation="tanh",
                        alpha=0.01,
                        hidden_layer_sizes=(2,),
                        max_iter=max_iter,
                        tol=tol)

    mlp.fit(X_train, y_train)

    # Работа с ненормализованными данными
    predictions = mlp.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("MLP: ", acc)
    acc_mlp.append(acc)

    # Работа с нормализованными данными
    mlp = MLPClassifier(random_state=i,
                        solver="sgd",
                        activation="tanh",
                        alpha=0.01,
                        hidden_layer_sizes=(2,),
                        max_iter=max_iter,
                        tol=tol)
    mlp.fit(X_train_scaled, y_train)
    predictions = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print("MLP with Norm: ", acc)
    acc_mlpn.append(acc)

    """
    The solver for weight optimization.

    ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.

    ‘sgd’ refers to stochastic gradient descent.

    ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

    Note: The default solver ‘adam’ works pretty well on relatively large datasets 
    (with thousands of training samples or more) in terms of both training time 
    and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.
    """

    # Работа с решателем lbfgs
    mlp = MLPClassifier(random_state=i,
                        solver="lbfgs",
                        activation="tanh",
                        alpha=0.01,  #
                        hidden_layer_sizes=(2,),
                        max_iter=max_iter,
                        tol=tol)
    mlp.fit(X_train_scaled, y_train)
    predictions = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print('Solver: lbfgs \t Activation: tanh')
    print("MLP with Norm: ", acc)
    acc_mlpn_lbfgs.append(acc)

    # Работа с функцией активации logistic
    mlp = MLPClassifier(random_state=i,
                        solver="lbfgs",
                        activation="logistic",
                        alpha=0.01,
                        hidden_layer_sizes=(2,),
                        max_iter=max_iter,
                        tol=tol)
    mlp.fit(X_train_scaled, y_train)
    predictions = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    print('Solver: lbfgs \t Activation: logistic')
    print("MLP with Norm: ", acc)
    acc_mlpn_logistic.append(acc)

    # Сравнение данных
    # print(X_test[:10])
    # print(X_test_scaled[:10])

# Распечатка итоговых результатов
print("Perceptron: ", min(acc_p), median(acc_p), max(acc_p), np.std(acc_p))
print("Perceptron with Norm: ", min(acc_pn), median(acc_pn), max(acc_pn),
      np.std(acc_pn))

print("MLP: ", min(acc_mlp), median(acc_mlp), max(acc_mlp), np.std(acc_mlp))
print("MLP with Norm: ", min(acc_mlpn), median(acc_mlpn), max(acc_mlpn),
      np.std(acc_mlpn))

# Расчет минимума и максимума для графика
X = np.concatenate((X_train, X_test), axis=0)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# Построение графика
figure = plt.figure(figsize=(17, 9))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, 1, 1)

# Точки из обучающей выборки
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)

# Тестовые точки
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
plt.show()


snsplot = sns.kdeplot(X_test, shade=True)
fig = snsplot.get_figure()

"""
График распределения - https://habr.com/ru/post/470535/

"""
