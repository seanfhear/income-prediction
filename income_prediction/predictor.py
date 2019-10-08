import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict

TRAINING_DATA_FILE = '../data/training_data.csv'
TEST_DATA_FILE = '../data/tcd_ml_2019-20_income_prediction_test_(without_labels).csv'
OUT_FILE = '../data/output.csv'

test_data = pd.read_csv(TEST_DATA_FILE)
test_data = pd.concat((test_data.drop(columns='Gender'), pd.get_dummies(test_data['Gender'])), axis=1)
test_data = pd.concat((test_data.drop(columns='University Degree'), pd.get_dummies(test_data['University Degree'])), axis=1)
test_data = pd.concat((test_data.drop(columns='Country'), pd.get_dummies(test_data['Country'])), axis=1)
test_data = pd.concat((test_data.drop(columns='Profession'), pd.get_dummies(test_data['Profession'])), axis=1)

train_data = pd.read_csv(TRAINING_DATA_FILE)
train_data = pd.concat((train_data.drop(columns='Gender'), pd.get_dummies(train_data['Gender'])), axis=1)
train_data = pd.concat((train_data.drop(columns='Degree'), pd.get_dummies(train_data['Degree'])), axis=1)
train_data = pd.concat((train_data.drop(columns='Country'), pd.get_dummies(train_data['Country'])), axis=1)
train_data = pd.concat((train_data.drop(columns='Profession'), pd.get_dummies(train_data['Profession'])), axis=1)

missing_cols = set(train_data.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0
test_data = test_data[train_data.columns]

dropped_test_cols = ["Instance", "Size of City", "Income in EUR"]
X_test = test_data.drop(dropped_test_cols, axis=1).values.reshape(-1, len(test_data.columns) - len(dropped_test_cols))

dropped_train_cols = ["Instance", "Income in EUR", "Size of City"]
X_train = train_data.drop(dropped_train_cols, axis=1).values.reshape(-1, len(train_data.columns) - len(dropped_train_cols))
y_train = train_data["Income in EUR"].values.reshape(-1, 1)

model = LinearRegression()

# kf = KFold(n_splits=5)
# kf.get_n_splits(X)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pd.DataFrame(y_pred).to_csv(OUT_FILE, header=None, index=None)
# out_df = pd.read_csv(TEST_DATA_FILE)
# out_df = out_df[["Instance"]]
# print(out_df)
# out_df = pd.concat(pd.DataFrame(y_pred), [out_df])
# print(out_df)
# out_df.to_csv(OUT_FILE, header=None, index=None)

# print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# plt.scatter(y_test, y_pred)
# plt.show()

# df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(df)

# scores = cross_val_score(model, X, y, cv=5)
# print("Cross-validated scores:", scores)