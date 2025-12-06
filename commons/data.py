import os
import pandas as pd

def load_data():
    """
    Loads data from the reacher3 dataset.
    :return: (X_train, Y_train, x_test, y_test)
    """
    # TODO: adattare per caricamento generico di datasets
    path = os.path.dirname(os.path.abspath(__file__))
    train_ds = pd.read_csv(path + '/../datasets/reacher3_train_1.csv')
    input_training = train_ds.iloc[:, :7].to_numpy()
    output_training = train_ds.iloc[:, -3:].to_numpy()

    test_ds = pd.read_csv(path + '/../datasets/reacher3_test_1.csv')
    input_test = test_ds.iloc[:, :7].to_numpy()
    output_test = test_ds.iloc[:, -3:].to_numpy()
    return input_training, output_training, input_test, output_test