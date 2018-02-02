from os.path import dirname, realpath
import pandas as pd
import numpy as np

dir_path = dirname(realpath(__file__))

# categorical data columns containing missing values
categorical_data = [
    'Smokes',
    'Hormonal Contraceptives',
    'STDs',
]

# continuous data columns containing missing values
continuous_data = [
    'Number of sexual partners',
    'First sexual intercourse',
    'Num of pregnancies',
    'Smokes (years)',
    'Smokes (packs/year)',
    'Hormonal Contraceptives (years)',
    'STDs (number)',
    'STDs:condylomatosis',
    'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:AIDS',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV',
    'STDs: Time since first diagnosis',
    'STDs: Time since last diagnosis',
]

# Treat this data differently because IUD (years)
# is dependent on IUD and so these rows need to carefully
# be filled together (or just not at all... so that's what we'll do)
special_data = [
    'IUD',
    'IUD (years)',
]

def fill_missing_data(data, columns, func):
    for entry in columns:
        data[entry] = func(data, entry)

    return data

def clean(dataset):
    data = dataset.replace('?', np.nan)
    data = data.convert_objects(convert_numeric=True)

    # fill missing data
    data = fill_missing_data(data, continuous_data, lambda d, k: d[k].fillna(d[k].median()))
    data = fill_missing_data(data, categorical_data, lambda d, k: d[k].fillna(1))
    data = fill_missing_data(data, special_data, lambda d, k: d[k].fillna(0))

    # for categorical data
    all_cat_cols = [
        'Smokes',
        'Hormonal Contraceptives',
        'IUD',
        'STDs', 'Dx:Cancer',
        'Dx:CIN',
        'Dx:HPV',
        'Dx',
        'Hinselmann',
        'Citology',
        'Schiller'
    ]
    data = pd.get_dummies(data=data, columns=all_cat_cols)

    return data

def get_X_Y(dataset):
    X_cols = list(dataset.columns.values)
    y_cols = [X_cols.pop(X_cols.index('Biopsy'))]

    np.random.seed(42)
    dataset_shuffle = dataset.iloc[np.random.permutation(len(dataset))]

    X = dataset_shuffle[X_cols]
    y = np.array(dataset_shuffle[y_cols])

    return X, y

from sklearn.cross_validation import train_test_split
def get_train_test_set():
    dataset = pd.read_csv(dir_path + '/../data/kag_risk_factors_cervical_cancer.csv')
    cleaned_data = clean(dataset)

    X, y = get_X_Y(cleaned_data)
    return train_test_split(X, y, test_size=0.20, random_state=0)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test_set()
