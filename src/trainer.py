from statistics import LinearRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os

def main():
    # load training and test data
    train_x, train_y, input_test, output_test = load_data()

    # preprocess data (normalization)

    # linear regression


def load_data() -> np.array:
    path = os.path.dirname(os.path.abspath(__file__))
    train_ds = pd.read_csv(path + '/../datasets/reacher3_train_1.csv')
    input_training = train_ds.iloc[:, :7].to_numpy()
    output_training = train_ds.iloc[:, -3:].to_numpy()

    test_ds = pd.read_csv(path + '/../datasets/reacher3_test_1.csv')
    input_test = test_ds.iloc[:, :7].to_numpy()
    output_test = test_ds.iloc[:, -3:].to_numpy()
    return input_training, output_training, input_test, output_test


class LinearTrainer:
    def __init__(self, model_name="reacher3_linear"):
        self.model_name = model_name
        self.model = LinearRegression()
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.is_trained = False

    def train(self, X_train, Y_train):
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)




if __name__ == "__main__":
    main()
