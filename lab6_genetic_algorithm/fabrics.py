import random

from lab6_genetic_algorithm.nutrition_entities import ProductCombination, \
    Product


def create_product_combination(all_products, product_combination_size, ethalon):
    """ Заполнение продуктового набора случайными продуктами из полного спска"""
    product_combination = ProductCombination(ethalon)

    # for i in range(product_combination_size):
    while len(product_combination) < product_combination_size:
        product = all_products[random.randint(0, len(all_products) - 1)]
        if product not in product_combination:
            product_combination.append(product)

    return product_combination


def create_product(name='Продукт', ethalon=None, product_combination_size=None, price_from=0, price_to=400):
    """ Создание продукта"""
    max_protein = (ethalon.protein / product_combination_size) * 2
    max_fats = (ethalon.fats / product_combination_size) * 2
    max_carbs = ethalon.carbs / product_combination_size * 2

    min_price = int(price_from / product_combination_size / 2)
    max_price = int(price_to / product_combination_size)

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


def clone(value):
    ind = ProductCombination(value.ethalon, value[:])
    return ind
