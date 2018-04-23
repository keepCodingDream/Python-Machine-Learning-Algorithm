# coding:UTF-8

"""
  @Date 2018-04-16
  @author tracy
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

Y_path = '../data/clean_y.csv'
X_path = '../data/clean_X.csv'
test_path = '../data/X_test.csv'
id_path = '../data/my_ids.csv'
country_path = '../data/my_country.csv'

X = pd.read_csv(X_path)
Y = pd.read_csv(Y_path)
ids = pd.read_csv(id_path)
country = pd.read_csv(country_path)
test_X = pd.read_csv(test_path)

# label final
le = LabelEncoder()
Y = le.fit_transform(Y.values)
# Xgboost
params = {"objective": "multi:softmax", "num_class": 12}
T_train_xgb = xgb.DMatrix(X, Y)
X_test_xgb = xgb.DMatrix(test_X)
gbm = xgb.train(params, T_train_xgb, 20)
Y_predicted = gbm.predict(X_test_xgb)
# back to label
Y_predicted = le.inverse_transform(Y_predicted.astype('int64'))
Y_predicted = pd.DataFrame(Y_predicted, columns=['country'])
X_size = Y_predicted.shape[0]
ids = ids[(ids.shape[0] - X_size):(ids.shape[0] + 1)]
pd.DataFrame(ids, columns=['id']).to_csv(path_or_buf='../data/finalIds.csv', columns=['id'], index=False)
ids = pd.read_csv('../data/finalIds.csv')
final = pd.concat([ids, Y_predicted], axis=1)
print final
pd.DataFrame(final, columns=['id', 'country']).to_csv(path_or_buf='../data/xgBoost_predict.csv', index=False)
