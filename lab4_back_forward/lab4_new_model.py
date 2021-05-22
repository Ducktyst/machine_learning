import numpy as np


# Определяем функцию активации
def activation(x: np.ndarray) -> np.ndarray:
    """
    Функция активации
    :param x: значения весов
    :return: результат применения функции активации
    к каждому входному значению нейрона
    """
    return (1 / (1 + np.exp(-x)))


# Определяем производную от функции активации
def sigma_derivative(x: float):
    return (x * (1 - x))


X = np.array([
    [0, 0, 1],
    [0.3, 1, 0],
    [1, 0.3, 0],
    [0.6, 0.2, 1],
    [0.6, 0.2, 1]
])

Y = np.array(
    [[0],
     [1],
     [1],
     [0],
     [1]]
)

np.random.seed(4)

W_1_2 = 2 * np.random.random((3, 2)) - 1
W_2_3 = 2 * np.random.random((2, 2)) - 1
W_3_4 = 2 * np.random.random((2, 1)) - 1

speed = 1.1  #

for jj in range(10000):

    l1 = X
    l2 = activation(np.dot(l1, W_1_2))
    l3 = activation(np.dot(l2, W_2_3))
    l4 = activation(np.dot(l3, W_3_4))
    l4_error = Y - l4

    if (jj % 100) == 0:
        # print("Errors: ", l4_error)
        print("Error: ", np.mean(np.abs(l4_error)))

    l4_sigma = l4_error * sigma_derivative(l4_error)
    # print(l4_sigma)

    l3_error = l4_sigma.dot(W_3_4.T)
    l3_sigma = l3_error * sigma_derivative(l3_error)

    # l2_error = l3_sigma.dot(W_2_3.T)

    l2_error = l3_sigma.dot(W_2_3.T)
    l2_sigma = l2_error * sigma_derivative(l2_error)

    W_3_4 = W_3_4 + speed * l3.T.dot(l4_sigma)
    W_2_3 = W_2_3 + speed * l2.T.dot(l3_sigma)
    W_1_2 = W_1_2 + speed * l1.T.dot(l2_sigma)

X_test = np.array(
    [[0, 0, 0],
     [0, 1, 1],
     [1, 0, 1],
     [1, 1, 0],
     [0.5, 0.5, 0],
     [0.5, 0.5, 1]]
)

l1 = X_test
l2 = activation(np.dot(l1, W_1_2))
l3 = activation(np.dot(l2, W_2_3))
l4 = activation(np.dot(l2, W_3_4))

# Y_test должен получиться [1, 0, 0, 1, 1, 0]
Y_test = l4
print(Y_test)
