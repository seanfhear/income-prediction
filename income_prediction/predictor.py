import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics

TRAINING_DATA_FILE = '../data/training3.csv'
TEST_DATA_FILE = '../data/tcd ml 2019-20 income prediction test (without labels).csv'
OUT_FILE = '../data/output.csv'
TRAINING_OUT_FILE = '../data/training_output.csv'

MISSING_VALUES = ['#N/A']
UNKNOWN_COLS = ['Profession', 'Country', 'Year of Record', 'Age']
STR_COLS = ['Gender', 'University Degree', 'Hair Color']
INT_COLS = ['Size of City', 'Wears Glasses', 'Body Height [cm]']

DUMMY_COLS = ['Gender', 'University Degree', 'Profession', 'Hair Color', 'Country']
COLS_TO_CLEAN = ['Gender', 'University Degree', 'Age']
COLS_TO_SCALE = ['Year of Record', 'Age', 'Size of City']

COLS_TO_CONVERT_SPARSE = ['Country', 'Profession']
LOW_FREQUENCY_THRESHOLD = 5

DROPPED_COLS = ['Instance']
TARGET_COLUMN = 'Income in EUR'

NUM_FOLDS = 5


def get_df_from_csv(filename, training):
    df = pd.read_csv(filename, na_values=MISSING_VALUES)
    df = clean_data(df, training)

    return df


def oh_encode(df):
    for col in DUMMY_COLS:
        df = pd.concat((df.drop(columns=col), pd.get_dummies(df[col], drop_first=True)), axis=1)
    return df


def binary_encode(df):
    ''


def clean_data(df, training):
    for col in STR_COLS:
        clean_str_col(df, col)

    for col in INT_COLS:
        clean_int_col(df, col)

    for col in UNKNOWN_COLS:
        df = remove_unknowns(df, col, training)

    df['Gender'] = df['Gender'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('none', 'unknown')
    df['Hair Color'] = df['Hair Color'].replace('0', 'unknown')

    # for col in COLS_TO_SCALE:
    #     normalize_col(df, col)

    df = remove_outliers(df, training)

    convert_sparse_values(df, COLS_TO_CONVERT_SPARSE, threshold=LOW_FREQUENCY_THRESHOLD)

    return df


def clean_str_col(df, col):
    df[col].fillna('unknown', inplace=True)


def clean_int_col(df, col):
    median = df[col].median()
    df[col].fillna(median, inplace=True)


def remove_unknowns(df, col, training):
    if training:
        df[col].fillna('unknown', inplace=True)
        df = df[df[col] != 'unknown']
    else:
        if col in ['Age', 'Year of Record']:
            clean_int_col(df, col)
        else:
            clean_str_col(df, col)
    return df


def remove_outliers(df, training):
    if training:
        df = df[df[TARGET_COLUMN] < 3000000]
    return df


def convert_sparse_values(df, cols, threshold, replacement='other'):
    for col in cols:
        counts = df[col].value_counts()
        to_convert = counts[counts <= threshold].index
        df[col] = df[col].replace(to_convert, replacement)


def normalize_col(df, col):
    max_value = df[col].max()
    min_value = df[col].min()
    df[col] = (df[col] - min_value) / (max_value - min_value)


def get_train_and_test():
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


def cross_val_train(df_train):
    x = df_train.drop([TARGET_COLUMN], axis=1).values.reshape(-1, len(df_train.columns) - 1)
    y = df_train[TARGET_COLUMN].values.reshape(-1, 1)

    kf = KFold(n_splits=NUM_FOLDS)
    kf.get_n_splits(x)

    rmse_sum = 0

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        df.to_csv(TRAINING_OUT_FILE)

        rmse_sum += np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print(rmse_sum / NUM_FOLDS)


def main_event(df_train, df_test):
    x_train = df_train.drop(TARGET_COLUMN, axis=1).values.reshape(-1, len(df_train.columns) - 1)
    y_train = df_train[TARGET_COLUMN].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_test = df_test.drop(TARGET_COLUMN, axis=1).values.reshape(-1, len(df_test.columns) - 1)
    y_pred = model.predict(x_test)

    df = pd.DataFrame({'Predicted': y_pred.flatten()})
    df.to_csv(OUT_FILE)


def main(train):
    train_data, test_data = get_train_and_test()
    if train:
        cross_val_train(train_data)
    else:
        main_event(train_data, test_data)


if __name__ == "__main__":
    main(train=1)
