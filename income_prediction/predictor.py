import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt


TRAINING_DATA_FILE = '../data/training4.csv'
TEST_DATA_FILE = '../data/tcd ml 2019-20 income prediction test (without labels).csv'
OUT_FILE = '../data/output.csv'
TRAINING_OUT_FILE = '../data/training_output.csv'

MISSING_VALUES = ['#N/A']
UNKNOWN_COLS = ['Age', 'Year of Record', 'Country', 'Profession']

DUMMY_COLS = ['Country', 'Gender', 'University Degree', 'Profession']

COLS_TO_CONVERT_SPARSE = ['Country', 'Profession']
LOW_FREQUENCY_THRESHOLD = 5

COLS_TO_TRANSFORM = ['Size of City']

DROPPED_COLS = ['Instance', 'Hair Color']
TARGET_COLUMN = 'Income in EUR'

NUM_FOLDS = 3


def get_df_from_csv(filename, training):
    """
    Read in a csv file as a dataframe and clean the data
    :param filename:
    :param training:
    :return:
    """
    df = pd.read_csv(filename, na_values=MISSING_VALUES)
    df = clean_data(df, training)
    return df


def oh_encode(df):
    """
    One Hot Encode the columns defined in DUMMY_COLS
    :param df:
    :return:
    """
    for col in DUMMY_COLS:
        df = pd.concat((df.drop(columns=col), pd.get_dummies(df[col], drop_first=True)), axis=1)
    return df


def process_professions(df, training):
    """
    Perform specific actions on the Profession feature
    :param df:
    :param training: Whether the data is the training set or not
    :return:
    """
    mean_salaries = df.groupby('Profession')['Income in EUR'].mean()
    # print(mean_salaries['Accounts Clerk'])

    remove_unknowns(df, 'Profession', training)
    le = LabelEncoder()
    df['Profession'] = le.fit_transform(df['Profession'])


def clean_data(df, training):
    """
    Cleans the data by unifying different types of unknowns, transforming columns defined
    in COLS_TO_TRANSFORM, removing outliers and merging values that occur infrequently
    :param df:
    :param training: Whether the data is the training set or not
    :return:
    """
    process_professions(df, training)
    df["Income in EUR"] = pd.to_numeric(df["Income in EUR"])

    for col in UNKNOWN_COLS:
        df = remove_unknowns(df, col, training)

    df['Gender'] = df['Gender'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('none', 'unknown')
    df['Hair Color'] = df['Hair Color'].replace('0', 'unknown')

    for col in COLS_TO_TRANSFORM:
        transform_col(df, col)

    df = remove_outliers(df, training)

    convert_sparse_values(df, COLS_TO_CONVERT_SPARSE, threshold=LOW_FREQUENCY_THRESHOLD)

    return df


def clean_str_col(df, col):
    """
    Replaces all 'unknown' values in a string column
    :param df:
    :param col:
    :return:
    """
    df[col].fillna('unknown', inplace=True)


def clean_num_col(df, col):
    """
    Replaces all 'unknown' values in an number column with the median value for that column
    :param df:
    :param col:
    :return:
    """
    median = df[col].median()
    df[col].fillna(median, inplace=True)


def remove_unknowns(df, col, training):
    """
    Replaces any type of NaN or empty value with 'unknown' and calls the cleaning functions
    :param df:
    :param col:
    :param training:
    :return:
    """
    if training:
        df[col].fillna('unknown', inplace=True)
        df = df[df[col] != 'unknown']
    else:
        if col in ['Age', 'Year of Record']:
            clean_num_col(df, col)
        else:
            clean_str_col(df, col)
    return df


def remove_outliers(df, training):
    """
    Removes entries in the dataset above a certain threshold
    :param df:
    :param training:
    :return:
    """
    if training:
        df = df[df[TARGET_COLUMN] < 4000000]
    return df


def convert_sparse_values(df, cols, threshold, replacement='other'):
    """
    Merges infrequent values into one 'other' value
    :param df:
    :param cols:
    :param threshold:
    :param replacement:
    :return:
    """
    for col in cols:
        counts = df[col].value_counts()
        to_convert = counts[counts <= threshold].index
        df[col] = df[col].replace(to_convert, replacement)


def transform_col(df, col):
    """
    Log transforms a column
    :param df:
    :param col:
    :return:
    """
    df[col] = df[col].apply(np.log)


def get_train_and_test():
    """
    Returns a dataframe of the training dataset and a dataframe of the test dataset
    :return:
    """
    df_train = get_df_from_csv(TRAINING_DATA_FILE, training=True)
    df_test = get_df_from_csv(TEST_DATA_FILE, training=False)

    # https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f
    df_train['train'] = 1
    df_test['train'] = 0

    for col in DROPPED_COLS:
        df_train = df_train.drop([col], axis=1)
        df_test = df_test.drop([col], axis=1)

    combined = pd.concat([df_train, df_test])
    combined = oh_encode(combined)

    df_train = combined[combined['train'] == 1]
    df_test = combined[combined['train'] == 0]
    df_train = df_train.drop(['train'], axis=1)
    df_test = df_test.drop(['train'], axis=1)

    return df_train, df_test


def cross_val_train(df_train, model):
    """
    Performs K-Fold cross validation testing of the model. Number of folds is defined in NUM_FOLDS.
    RMSE is calculated for each fold with the mean RMSE also output. Produces a graph which plots
    predictions against actual values.
    :param df_train:
    :param model:
    :return:
    """
    x = df_train.drop([TARGET_COLUMN], axis=1).values.reshape(-1, len(df_train.columns) - 1)
    y = df_train[TARGET_COLUMN].values.reshape(-1, 1)

    kf = KFold(n_splits=NUM_FOLDS)
    kf.get_n_splits(x)

    rmse_sum = 0

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train.ravel())

        y_pred = model.predict(x_test)

        df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        df.to_csv(TRAINING_OUT_FILE)

        rmse_sum += np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('y_test')
        ax.set_ylabel('y_pred')
        plt.plot([-50000, 2500000], [-50000, 2500000], color='red', linewidth=2)
        plt.xlim(-150000, 3000000)
        plt.ylim(-150000, 3000000)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    print(rmse_sum / NUM_FOLDS)


def get_predictions(df_train, df_test, model):
    """
    Trains a model and uses it to make predictions for the test dataset. Predictions are output to CSV.
    :param df_train:
    :param df_test:
    :param model:
    :return:
    """
    x_train = df_train.drop(TARGET_COLUMN, axis=1).values.reshape(-1, len(df_train.columns) - 1)
    y_train = df_train[TARGET_COLUMN].values.reshape(-1, 1)

    model.fit(x_train, y_train)

    x_test = df_test.drop(TARGET_COLUMN, axis=1).values.reshape(-1, len(df_test.columns) - 1)
    y_pred = model.predict(x_test)

    df = pd.DataFrame({'Predicted': y_pred.flatten()})
    df.to_csv(OUT_FILE)


def main(train):
    """
    Runs cross_val_train or get_predictions based on the boolean passed in.
    :param train:
    :return:
    """
    train_data, test_data = get_train_and_test()
    model = LinearRegression()
    # model = SGDRegressor(
    #     loss="squared_loss",
    #     penalty="elasticnet",
    #     alpha=0.000001
    # )
    if train:
        cross_val_train(train_data, model)
    else:
        get_predictions(train_data, test_data, model)


if __name__ == "__main__":
    main(train=1)
