import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x - 5 * x + 5


def df(x):
    return 2 * x - 5


N = 10  # число итераций
xx = 0  # начальное значение
lmd = 0.1 # шаг сходимости1

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()  # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()  # создание окна и осей для графика
ax.grid(True)  # отображение сетки на графике

ax.plot(x_plt, f_plt)  # отображение параболы
point = ax.scatter(xx, f(xx), c='red')  # отображение точки красным цветом

for i in range(N):
    dfxx = df(xx)
    delta = lmd*dfxx
    print(xx, dfxx, delta)

    xx = xx - delta # изменение агрумента на текущей итерации


    point.set_offsets([xx, f(xx)]) # отображение нового положения точки

    # перерисовка графика и задержка на 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()



