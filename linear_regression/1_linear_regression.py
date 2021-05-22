from  sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model.Ridge
from sklearn.feature_extraction import DictVectorizer


vectorizer = TfidfVectorizer(min_df=10)
train_text_feature_matrix = vectorizer.fit_transform(data_train['FullDescription'])
idf = vectorizer.idf_
print(dict(zip(vectorizer.get_feature_names(), idf)))



enc = DictVectorizer()
train_dic = data_train[['LocationNormalized', 'ContractTime']].to_dict('records')
test_dic = data_test[['LocationNormalized', 'ContractTime']].to_dict('records')
X_train_categ = enc.fit_transform(train_dic)
X_test_categ = enc.transform(test_dic)

data_train['LocationNormalized'].fillna('nan', inplace=True)


# стр. 193

# https://drive.google.com/drive/folders/0B5yyS8oSQ0FDelpKRXg3c3lKVlU