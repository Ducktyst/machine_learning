import random
import matplotlib.pyplot as plt

"""
Дано N наименовний продуктов, для каждого из которых 
известно m характеристик. Необходимо получить самый лучший по 
характеристикам рацион из k наименований, удовлетворяющий заданным
 ценовым рамкам. Лучшим считается рацион с минимальным отклонением от нормы

N - размер популяции 
m - количество характеристик продукта

k - длина битовой строки, где каждому биту соответсвтвует продукт 
    k - количество активных генов в наборе?

цена - одна из характеристик
"""

# константы задачи
TOTAL_PRODUCTS_COUNT = 400  # N - ассортимент продуктов

# константы генетического алгоритма
PRODUCT_COMBINATION_SIZE = 7  # количество индивидуумов в популяции (количество комбинаций продуктов == кол-во хромосом, допустимых решений)
P_CROSSOVER = 0.2  # вероятность скрещивания
P_MUTATION = 0  # вероятность мутации индивидуума

POPULATION_SIZE = 50  # количество особей (продуктовых наборов в популяции)

MAX_GENERATIONS = 100  # максимальное количество поколений (количество итераций)
P_SURVIVE = 0.8

RANDOM_SEED = 43
random.seed(RANDOM_SEED)

PRICE_FROM = 200
PRICE_TO = 5000


class Product:
    # Возможно стоит преобразовать в список характеристик, а значения получать по индексам
    def __init__(self, name, price, protein, fats, carbs):
        self.name = name
        self.price = price
        self.protein = protein
        self.fats = fats
        self.carbs = carbs

    def __str__(self):
        return f'{self.name} Цена: {self.price} Б: {self.protein} Ж: {self.fats} У: {self.carbs}'


class ProductCombination(list):

    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = self.calc_fitness()
        self.total_price = self.calc_total_price()

        self.total_nutrition = Product('', 0, 0, 0, 0)
        for product in self:
            self.total_nutrition.protein += product.protein
            self.total_nutrition.fats += product.fats
            self.total_nutrition.carbs += product.carbs

    def calc_total_price(self):
        """ Вычисляет суммарную цену продуктового набора"""
        total_price = 0
        for product in self:
            total_price += product.price
        return 400
        return total_price

    def calc_fitness(self):
        """ Расчет приспособленности особи

        Вычисляется суммарное отклонение каждого продукта от идеального

        Чем ближе результат вычислений к нулю, тем более приспособлена особь
        """
        ethalon = ETHALON_PRODUCT_COMBINATION

        fitness = 0
        self.total_nutrition = Product('', 0, 0, 0, 0)
        for product in self:
            self.total_nutrition.protein += product.protein
            self.total_nutrition.fats += product.fats
            self.total_nutrition.carbs += product.carbs

        diff = \
            abs(ethalon.protein - self.total_nutrition.protein) + \
            abs(ethalon.fats - self.total_nutrition.fats) + \
            abs(ethalon.carbs - self.total_nutrition.carbs)

        # чем больше отклонение, тем меньше приспособленность
        return diff

    def is_fit_in_price(self, price_from=PRICE_FROM, price_to=PRICE_TO):
        if price_from < self.calc_total_price() < price_to:
            return True
        return False

    def __contains__(self, item):
        for product in self:
            if product == item:
                return True
        return False

    def append(self, object):
        res = super().append(object)
        self.fitness = self.calc_fitness()
        self.total_price = self.calc_total_price()

        self.total_nutrition = Product('', 0, 0, 0, 0)
        for product in self:
            self.total_nutrition.protein += product.protein
            self.total_nutrition.fats += product.fats
            self.total_nutrition.carbs += product.carbs
        return res

    def __str__(self):
        return f'Цена: {self.calc_total_price()} ' \
            f'Б: {self.total_nutrition.protein} ' \
            f'Ж: {self.total_nutrition.fats} ' \
            f'У: {self.total_nutrition.carbs} '\
            f'Отклонение: {self.calc_fitness()}'


ETHALON_PRODUCT_COMBINATION = Product(
    name='Целенвые значения нутриентов',
    price=1,
    protein=200 * 4,
    fats=100 * 9,
    carbs=200 * 4,
)


def productCombinationCreator(all_products, product_combination_size):
    """ Заполнение продуктового набора случайными продуктами из полного спска"""
    product_combination = ProductCombination()

    for product in all_products:
        if len(product_combination) == product_combination_size:
            break

        if random.random() > 0.8:
            product_combination.append(product)

    while len(product_combination) < product_combination_size:
        product = all_products[random.randint(0, TOTAL_PRODUCTS_COUNT - 1)]
        if product not in product_combination:
            product_combination.append(product)

    return product_combination


def clone(value):
    ind = ProductCombination(value[:])
    return ind


def selTournament(population, p_len):
    """
    Формирует новую популяцию из продуктовых наборов

    :param population: исходная популяия
    :param p_len: длина новой популяции
    :return:
    """
    offspring = []
    """"""
    for n in range(p_len):
        pc1_idx = random.randint(0, int((len(population) - 1) / 2))
        product_combination_1 = population[pc1_idx]
        pc2_idx = random.randint(int((len(population) - 1) / 2),
                                 len(population) - 1)
        product_combination_2 = population[pc2_idx]

        while not product_combination_1.is_fit_in_price():
            product_combination_1 = \
                population[random.randint(0, len(population) - 1)]

        while not product_combination_2.is_fit_in_price():
            product_combination_2 = \
                population[random.randint(0, len(population) - 1)]

        if random.random() >= P_SURVIVE:  # выживает сильнейшая особь
            strognest = max([product_combination_1, product_combination_2],
                            key=lambda pc: product_combination_1.calc_fitness())
            offspring.append(strognest)
        else:
            weakest = min([product_combination_1, product_combination_2],
                          key=lambda pc: pc.calc_fitness())
            offspring.append(weakest)

    return offspring


def mutate_product_combination(mutant, all_products, indpb=0.01):
    assert isinstance(mutant, ProductCombination)
    mutant_len = len(mutant)

    for product in all_products:  # не гарантирует, что будет изменено именно 4 продукта
        # ограничивает...
        if random.random() < indpb:
            if product in mutant:
                mutant.remove(product)
            else:
                mutant.append(product)


def mutate_product_combination_save_len(mutant, all_products, indpb=0.01):
    """ Мутация одного гена, при сохранении количества продуктов"""
    assert isinstance(mutant, ProductCombination)
    mutant_len = len(mutant)

    mutate_products_count = 4  # количество измееенных продуктов
    while mutate_products_count > 0:
        for product in all_products:  # не гарантирует, что будет изменено именно 4 продукта
            # ограничивает...
            if random.random() < indpb:
                if product in mutant:
                    if mutate_products_count == 0:
                        break
                    mutant.remove(product)
                    mutate_products_count += 1
                else:
                    if mutate_products_count == 0:
                        break
                    mutant.append(product)
                    mutate_products_count -= 1

    assert len(mutant) == mutant_len


def tournament_selection(population, p_len):
    offspring = population[:]

    while len(offspring) > p_len:
        product_combination_1 = product_combination_2 = None
        while product_combination_1 == product_combination_2:
            pc1_idx = random.randint(0, int((len(offspring) - 1) / 2))
            product_combination_1 = offspring[pc1_idx]

            pc2_idx = random.randint(int((len(offspring) - 1) / 2),
                                     len(offspring) - 1)
            product_combination_2 = offspring[pc2_idx]

        weakest = max([product_combination_1, product_combination_2],
                      key=lambda pc: pc.calc_fitness())
        # weakest = max(offspring, key=lambda pc: pc.calc_fitness())
        # print(weakest.calc_fitness())
        try:
            offspring.remove(weakest)
        except ValueError:
            pass

    return offspring


def crossover(product_combination1, product_combination2, all_products):
    crossover_point = random.randint(1, PRODUCT_COMBINATION_SIZE - 1 - 2)

    new_product_combination = ProductCombination()

    # первая часть родителя1 и вторая часть родителя2
    for i in range(PRODUCT_COMBINATION_SIZE):
        product = all_products[i]

        if i <= crossover_point:
            product = all_products[i]

            if product in product_combination1:
                new_product_combination.append(product)

        elif i > crossover_point:
            if product in product_combination2:
                new_product_combination.append(product)

        if len(new_product_combination) == PRODUCT_COMBINATION_SIZE:
            break

    return new_product_combination


def crossover_2p(product_combination1, product_combination2, all_products):
    p1 = int(len(all_products) / 3)
    p2 = int(len(all_products) / 3 * 2)

    new_product_combination = ProductCombination()

    for i in range(len(all_products)):
        product = all_products[i]
        if p1 <= i <= p2:
            if product in product_combination2:
                new_product_combination.append(product)
            else:
                try:
                    new_product_combination.remove(product)
                except ValueError:
                    pass
        else:
            if product in product_combination1:
                new_product_combination.append(product)
            else:
                try:
                    new_product_combination.remove(product)
                except ValueError:
                    pass

    if len(new_product_combination) < PRODUCT_COMBINATION_SIZE:
        pass

    if len(new_product_combination) > PRODUCT_COMBINATION_SIZE:
        raise Exception("Превышено кол-во")
    return new_product_combination


def crossover_2p(product_combination1, product_combination2, all_products):
    p1 = int(len(all_products) / 3)
    p2 = int(len(all_products) / 3 * 2)

    new_product_combination = ProductCombination()

    for i in range(len(all_products)):
        product = all_products[i]
        if p1 <= i <= p2:
            if product in product_combination2:
                new_product_combination.append(product)
            else:
                try:
                    new_product_combination.remove(product)
                except ValueError:
                    pass
        else:
            if product in product_combination1:
                new_product_combination.append(product)
            else:
                try:
                    new_product_combination.remove(product)
                except ValueError:
                    pass

    # if len(new_product_combination) > PRODUCT_COMBINATION_SIZE:
    #     raise Exception("Превышено кол-во")
    assert isinstance(new_product_combination, ProductCombination)
    return new_product_combination


def productCreator(name='Продукт'):
    """ Создание продукта"""
    max_protein = (
                          ETHALON_PRODUCT_COMBINATION.protein / PRODUCT_COMBINATION_SIZE) * 2
    max_fats = (ETHALON_PRODUCT_COMBINATION.fats / PRODUCT_COMBINATION_SIZE) * 2

    min_carbs = int(
        ETHALON_PRODUCT_COMBINATION.carbs / PRODUCT_COMBINATION_SIZE / 2)
    max_carbs = int(
        ETHALON_PRODUCT_COMBINATION.carbs / PRODUCT_COMBINATION_SIZE * 2)

    min_price = int(PRICE_FROM / PRODUCT_COMBINATION_SIZE / 2)
    max_price = int(PRICE_TO / PRODUCT_COMBINATION_SIZE)
    price = random.randint(min_price, max_price)

    proteins = random.randint(1, int(max_protein))
    fats = random.randint(1, int(max_fats))
    carbs = random.randint(1, int(max_carbs))

    product = Product(
        name=name,
        price=price,
        protein=proteins,
        fats=fats,
        carbs=carbs,
    )

    return product


all_products = [productCreator(f'Продукт {i}') for i in
                range(TOTAL_PRODUCTS_COUNT)]


def populationCreator(population_size, all_products, product_combination_size):
    return [productCombinationCreator(all_products, product_combination_size)
            for i in range(population_size)]


population = populationCreator(population_size=POPULATION_SIZE * 2,
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
            child = crossover_2p(parent1, parent2, all_products)
            population.append(child)

    for mutant in population:
        if random.random() < P_MUTATION:
            mutant_copy = ProductCombination(mutant.copy())
            mutant_idx = population.index(mutant)
            mutate_product_combination(mutant_copy, all_products)
            population[mutant_idx] = mutant_copy

    offspring = tournament_selection(population, POPULATION_SIZE)
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
    print(
        f"Лучший продуктовый набор = {population[best_index]}\n")

print(ETHALON_PRODUCT_COMBINATION)

plt.plot(best_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность \n(отклонение от целевых значений)')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
