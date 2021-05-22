import random

from lab6_genetic_algorithm.nutrition_entities import ProductCombination


def mutate_product_combination(mutant, all_products, ethalon, indpb=0.05):
    """ Заменяет случайные продукты в наборе"""
    assert isinstance(mutant, ProductCombination)
    mutant_len = len(mutant)
    mutant = ProductCombination(ethalon, mutant[:])

    for product in all_products:
        if random.random() < indpb:
            if product in mutant:
                mutant.remove(product)
            else:
                mutant.append(product)

    while len(mutant) != mutant_len:
        if len(mutant) > mutant_len:
            mutant.remove(mutant[random.randint(0, len(mutant) - 1)])
        elif len(mutant) < mutant_len:
            product = all_products[random.randint(0, len(all_products) - 1)]
            mutant.append(product)

    assert len(mutant) == mutant_len
    return mutant


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


def crossover(product_combination1, product_combination2,
              product_combination_size, all_products, ethalon):
    crossover_point = random.randint(1, product_combination_size - 1 - 2)

    new_product_combination = ProductCombination(ethalon)

    # первая часть родителя1 и вторая часть родителя2
    for i in range(product_combination_size):
        product = all_products[i]

        if i <= crossover_point:
            product = all_products[i]

            if product in product_combination1:
                new_product_combination.append(product)

        elif i > crossover_point:
            if product in product_combination2:
                new_product_combination.append(product)

        if len(new_product_combination) == product_combination_size:
            break

    return new_product_combination


def crossover_2p(product_combination1, product_combination2, all_products,
                 ethalon):
    p1 = int(len(all_products) / 3)
    p2 = int(len(all_products) / 3 * 2)

    new_product_combination = ProductCombination(ethalon)

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

    assert isinstance(new_product_combination, ProductCombination)
    return new_product_combination


def selTournament(population, p_len, p_survive=0.05):
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

        if random.random() >= p_survive:  # выживает сильнейшая особь
            strognest = max([product_combination_1, product_combination_2],
                            key=lambda pc: product_combination_1.calc_fitness())
            offspring.append(strognest)
        else:
            weakest = min([product_combination_1, product_combination_2],
                          key=lambda pc: pc.calc_fitness())
            offspring.append(weakest)

    return offspring


def ranged_selection(population, p_len):
    """ Ранговый отбор"""
    offspring = population[:]

    while len(offspring) > p_len:
        weakest = max(offspring,key=lambda pc: pc.calc_fitness())
        offspring.remove(weakest)


    return offspring



def tournament_selection(population, p_len):
    offspring = population[:]

    while len(offspring) > p_len:
        pc1_idx = random.randint(0, len(offspring))
        product_combination_1 = offspring[pc1_idx]

        pc2_idx = random.randint(0, len(offspring))
        product_combination_2 = offspring[pc2_idx]

        weakest = max([product_combination_1, product_combination_2],
                      key=lambda pc: pc.calc_fitness())

        try:
            offspring.remove(weakest)
        except ValueError:
            pass

    return offspring

