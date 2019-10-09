import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

TRAINING_DATA_FILE = '../data/training.csv'
TEST_DATA_FILE = '../data/tcd ml 2019-20 income prediction test (without labels).csv'
OUT_FILE = '../data/output.csv'
TRAINING_OUT_FILE = '../data/training_output.csv'

MISSING_VALUES = ['#N/A']
STR_COLS = ['Gender', 'Profession', 'University Degree', 'Hair Color', 'Country']
INT_COLS = ['Year of Record', 'Age', 'Size of City', 'Wears Glasses', 'Body Height [cm]']

DUMMY_COLS = ['Gender', 'University Degree', 'Profession', 'Hair Color', 'Country']
IGNORED_COLS = ['Income in EUR']
COLS_TO_CLEAN = ['Gender', 'University Degree', 'Age']

DROPPED_COLS = []


def get_df_from_csv(filename):
    """
    Takes in a path to a csv file and returns a dataframe. Creates dummy variables for columns defined in DUMMY_COLS.
    :param filename: path to csv file
    :return: dataframe
    """
    df = pd.read_csv(filename, na_values=MISSING_VALUES)
    clean_data(df)
    return df


def encode_df(df):
    for col in DUMMY_COLS:
        df = pd.concat((df.drop(columns=col), pd.get_dummies(df[col], drop_first=True)), axis=1)
    return df


def clean_data(df):
    for col in STR_COLS:
        clean_str_col(df, col)

    for col in INT_COLS:
        clean_int_col(df, col)

    df['Gender'] = df['Gender'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('0', 'unknown')
    df['University Degree'] = df['University Degree'].replace('none', 'unknown')
    df['Hair Color'] = df['Hair Color'].replace('0', 'unknown')


def clean_str_col(df, col):
    df[col].fillna('unknown', inplace=True)


def clean_int_col(df, col):
    median = df[col].median()
    df[col].fillna(median, inplace=True)


def get_train_and_test():
    df_train = get_df_from_csv(TRAINING_DATA_FILE)
    df_test = get_df_from_csv(TEST_DATA_FILE)

    # https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f
    df_train['train'] = 1
    df_test['train'] = 0

    for col in DROPPED_COLS:
        df_train = df_train.drop([col], axis=1)
        df_test = df_test.drop([col], axis=1)

    combined = pd.concat([df_train, df_test])
    combined = encode_df(combined)

    df_train = combined[combined['train'] == 1]
    df_test = combined[combined['train'] == 0]
    df_train = df_train.drop(['train'], axis=1)
    df_test = df_test.drop(['train'], axis=1)

    return df_train, df_test


def train_and_test(df_train):
    x = df_train.drop(['Income in EUR'], axis=1).values.reshape(-1, len(df_train.columns) - len(IGNORED_COLS))
    y = df_train['Income in EUR'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df.to_csv(TRAINING_OUT_FILE)


def cross_val_train(df_train):
    x = df_train.drop(['Income in EUR'], axis=1).values.reshape(-1, len(df_train.columns) - len(IGNORED_COLS))
    y = df_train['Income in EUR'].values.reshape(-1, 1)

    model = LinearRegression()
    scores = cross_val_score(model, x, y, cv=5)
    print(scores)
    for i, score in enumerate(scores):
        print('{}: {}'.format(i+1, np.sqrt(score * -1)))


def main_event(df_train, df_test):
    x_train = df_train.drop(IGNORED_COLS, axis=1).values.reshape(-1, len(df_train.columns) - len(IGNORED_COLS))
    y_train = df_train["Income in EUR"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_test = df_test.drop(IGNORED_COLS, axis=1).values.reshape(-1, len(df_train.columns) - len(IGNORED_COLS))
    y_pred = model.predict(x_test)

    df = pd.DataFrame({'Predicted': y_pred.flatten()})
    df.to_csv(OUT_FILE)


train, test = get_train_and_test()

# train_and_test(train)
# cross_val_train(train)
main_event(train, test)
