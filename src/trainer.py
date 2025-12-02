from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import joblib
import pandas as pd
import numpy as np
import os

def main():
    trainer = LinearTrainer()
    # load training and test data
    train_x, train_y, test_x, test_y = load_data()

    # preprocess data (normalization)
    trainer.train(train_x, train_y)

    # linear regression
    trainer.evaluate(test_x, test_y)


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
        print("Data normalization...")
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)
        print("Done!")

        print("Model fitting...")
        self.model.fit(X_scaled, Y_scaled)
        self.is_trained = True
        print("Done!")

    def evaluate(self, X_test, Y_test) -> float:
        if not self.is_trained:
            print("ERROR: model not trained!")
            return
        # normalize test data
        X_test_scaled = self.scaler_X.transform(X_test)
        # predict from normalized data
        Y_pred_scaled = self.model.predict(X_test_scaled)
        # invert the transform
        y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # mean squared error
        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)

        print("Results:")
        print("MSE: %f" % mse)
        print("SCORE: %f" % r2)
        return y_pred

    def save_checkpoint(self, folder="checkpoints"):
        print("Saving checkpoint...")
        if not os.path.exists(folder):
            os.makedirs(folder)

        joblib.dump(self.model, folder + '/' + self.model_name + '_model.pkl')
        joblib.dump(self.scaler_X, folder + '/' + self.model_name + '_scalerX.pkl')
        joblib.dump(self.scaler_Y, folder + '/' + self.model_name + '_scalerY.pkl')
        print("Done!")

    def load_checkpoint(self, folder="checkpoints"):
        print("Loading checkpoint...")
        try:
            self.model = joblib.load(folder + '/' + self.model_name + '_model.pkl')
            self.scaler_X = joblib.load(folder + '/' + self.model_name + '_scalerX.pkl')
            self.scaler_Y = joblib.load(folder + '/' + self.model_name + '_scalerY.pkl')
            self.is_trained = True
            print("Done!")
        except FileNotFoundError:
            print("ERROR: no saved checkpoint found!")

if __name__ == "__main__":
    main()
