import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

TRAINING_DATA_FILE = '../data/training.csv'

def plot_feature_against_income(feature, sort_by_income=False):
    data = pd.read_csv(TRAINING_DATA_FILE)
    feature_data = data[[feature, "Income in EUR"]]

    feature_data = feature_data.sort_values([feature]) if not sort_by_income else feature_data.sort_values(["Income in EUR"])
    scaler = MinMaxScaler()
    feature_data['Income in EUR'] = scaler.fit_transform(feature_data[['Income in EUR']])
    if feature == "Size of City":
        feature_data[feature] = scaler.fit_transform(feature_data[[feature]])

    #feature_data.set_index(feature, inplace=True)

    #feature_data.plot.scatter(x=feature, y="Income in EUR")
    #feature_data.plot()

    feature_data.plot(x=feature, y='Income in EUR', style='o')
    plt.xscale('log')
    plt.show()

plot_feature_against_income("Size of City")
