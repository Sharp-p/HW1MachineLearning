import os
import pandas as pd

def load_data(N: int = 3):
    """
    Loads data from the reacherN dataset.
    :return: (X_train, Y_train, x_test, y_test)
    """
    path = os.path.dirname(os.path.abspath(__file__))
    # if it is planar (r3)
    if N == 3:
        train_ds = pd.read_csv(path + '/../datasets/reacher3_train_1.csv')
        input_training = train_ds.iloc[:, :7].to_numpy()
        output_training = train_ds.iloc[:, -3:].to_numpy()

        test_ds = pd.read_csv(path + '/../datasets/reacher3_test_1.csv')
        input_test = test_ds.iloc[:, :7].to_numpy()
        output_test = test_ds.iloc[:, -3:].to_numpy()
    else:
        #if it is spatial
        train_ds = pd.read_csv(path + f'/../datasets/reacher{N}_train_1.csv')
        input_training = train_ds.iloc[:, :N+6].to_numpy()
        output_training = train_ds.iloc[:, -N:].to_numpy()

        test_ds = pd.read_csv(path + f'/../datasets/reacher{N}_test_1.csv')
        input_test = test_ds.iloc[:, :N+6].to_numpy()
        output_test = test_ds.iloc[:, -N:].to_numpy()

    return input_training, output_training, input_test, output_test


