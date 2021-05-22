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

    def __init__(self, ethalon, *args):
        super().__init__(*args)
        self.ethalon = ethalon

        self.fitness = self.calc_fitness()
        self.total_price = self.calc_total_price()


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

        self.total_nutrition = Product('', 0, 0, 0, 0)
        for product in self:
            self.total_nutrition.protein += product.protein
            self.total_nutrition.fats += product.fats
            self.total_nutrition.carbs += product.carbs

        diff = \
            abs(self.ethalon.protein - self.total_nutrition.protein) + \
            abs(self.ethalon.fats - self.total_nutrition.fats) + \
            abs(self.ethalon.carbs - self.total_nutrition.carbs)

        # чем больше отклонение, тем меньше приспособленность
        return diff


    def is_fit_in_price(self, price_from=10, price_to=100):
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
            f'У: {self.total_nutrition.carbs} ' \
            f'Отклонение: {self.calc_fitness()}'
