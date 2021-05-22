import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

#Загрузка данных
data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

# Приведение текста к нижнему регистру
data_train['FullDescription'] = data_train.FullDescription.str.lower()
data_test['FullDescription'] = data_test.FullDescription.str.lower()

# Удаление ненужных символов
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ',
regex=True)

# Преобразование текста в вектор признаков, используя TdifVectorizer из sklearn
vectorizer = TfidfVectorizer(min_df=10)
train_text_feature_matrix = vectorizer.fit_transform(data_train['FullDescription'])
test_text_feature_matrix = vectorizer.transform(data_test['FullDescription'])

# Заполнение пустых ячеек
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

# Преобразование категориальных признаков в числовые
enc = DictVectorizer()
train_dic = data_train[['LocationNormalized', 'ContractTime']].to_dict('records')
test_dic = data_test[['LocationNormalized', 'ContractTime']].to_dict('records')
X_train_categ = enc.fit_transform(train_dic)
X_test_categ = enc.transform(test_dic)

# Здесь по горизонтали объединяется разреженная матрица с признакими текста (из
# столбца FullDescription) и матрица с закодированными категориями (из столбцов
# Location Normalized и Contract Time)
x_train = hstack((train_text_feature_matrix, X_train_categ))
y_train = data_train['SalaryNormalized'].values
x_test = hstack((test_text_feature_matrix, X_test_categ))

# Создание и обучение регрессии
ridge_regression = Ridge(alpha=1, random_state=241)
ridge_regression.fit(x_train, y_train)

# Получение и распечатка прогнозных значений
y_test = ridge_regression.predict(x_test)
print(y_test)