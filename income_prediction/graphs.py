import matplotlib.pyplot as plt
from income_prediction import predictor

TRAINING_DATA_FILE = '../data/training3.csv'


def scatter_plot(df, col):
    """
    Shows a scatter plot with Income on the y-axis and the passed in feature on the x-axis
    :param df:
    :param col:
    :return:
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df[col], df['Income in EUR'])
    ax.set_xlabel(col)
    ax.set_ylabel('Income in EUR')
    plt.show()


df_train, _ = predictor.get_train_and_test()
scatter_plot(df_train, 'Profession')
