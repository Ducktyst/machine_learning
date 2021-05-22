import random
from typing import NamedTuple

import matplotlib.pyplot as plt

from lab6_genetic_algorithm.fabrics import create_product_combination
from lab6_genetic_algorithm.fabrics import create_product
from lab6_genetic_algorithm.genetic import crossover_2p, tournament_selection, \
    mutate_product_combination, ranged_selection
from lab6_genetic_algorithm.nutrition_entities import Product

"""
Дано N наименовний продуктов, для каждого из которых 
известно m характеристик. Необходимо получить самый лучший по 
характеристикам рацион из k наименований, удовлетворяющий заданным
 ценовым рамкам. Лучшим считается рацион с минимальным отклонением от нормы
"""

# константы задачи
TOTAL_PRODUCTS_COUNT = 400  # N - ассортимент продуктов

# константы генетического алгоритма
PRODUCT_COMBINATION_SIZE = 7  # количество индивидуумов в популяции (количество комбинаций продуктов == кол-во хромосом, допустимых решений)
P_CROSSOVER = 0.5  # вероятность скрещивания
P_MUTATION = 0.5  # вероятность мутации индивидуума

POPULATION_SIZE = 50  # количество особей (продуктовых наборов в популяции)

MAX_GENERATIONS = 100  # максимальное количество поколений (количество итераций)
P_SURVIVE = 0.8

RANDOM_SEED = 40
random.seed(RANDOM_SEED)

PRICE_FROM = 200
PRICE_TO = 5000

ETHALON_PRODUCT_COMBINATION = Product(
    name='Целенвые значения нутриентов',
    price=1,
    protein=200 * 4,
    fats=100 * 9,
    carbs=200 * 4,
)

all_products = [create_product(f'Продукт {i}', ETHALON_PRODUCT_COMBINATION,
                               PRODUCT_COMBINATION_SIZE, PRICE_FROM, PRICE_TO)
                for i in range(TOTAL_PRODUCTS_COUNT)]


def populationCreator(population_size, all_products, product_combination_size):
    return [create_product_combination(all_products, product_combination_size,
                                       ETHALON_PRODUCT_COMBINATION)
            for i in range(population_size)]


population = populationCreator(population_size=POPULATION_SIZE,
                               all_products=all_products,
                               product_combination_size=PRODUCT_COMBINATION_SIZE)

fitness_values = [product_combination.calc_fitness()
                  for product_combination in population]

best_fitness_values = []  # список максимальных уровней приспособленности для поколений
mean_fitness_values = []  # список среднего уровня приспособленности каждого поколения
generationCounter = 0
while generationCounter < MAX_GENERATIONS:
    generationCounter += 1

    for parent1, parent2 in zip(population[::2], population[1::2]):
        if random.random() < P_CROSSOVER:
            child = crossover_2p(parent1, parent2, all_products,
                                 ETHALON_PRODUCT_COMBINATION)
            population.append(child)

    for mutant in population:
        if random.random() < P_MUTATION:
            mutant_idx = population.index(mutant)
            mutated = mutate_product_combination(mutant, all_products,
                                                 ETHALON_PRODUCT_COMBINATION)
            population.append(mutated)

    offspring = ranged_selection(population, POPULATION_SIZE)
    # offspring = tournament_selection(population, POPULATION_SIZE)
    offspring = sorted(offspring, key=lambda pc: pc.calc_fitness(),
                       reverse=True)
    population = offspring[:]

    fitness_values = [product_combination.calc_fitness()
                      for product_combination in population]
    best_fitness = min(fitness_values)
    mean_fitness = sum(fitness_values) / len(population)

    best_fitness_values.append(best_fitness)
    mean_fitness_values.append(mean_fitness)

    print(
        f"Поколение {generationCounter}: Макс приспособ. = {best_fitness},"
        f" Средняя приспособ.= {mean_fitness}")

    best_index = fitness_values.index(min(fitness_values))
    best_ind = population[fitness_values.index(min(fitness_values))]
    print(f"Лучший продуктовый набор = {population[best_index]}\n")

print(ETHALON_PRODUCT_COMBINATION)

plt.plot(best_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность \n(отклонение от целевых значений)')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
