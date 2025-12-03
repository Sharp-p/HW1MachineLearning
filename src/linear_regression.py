import os

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class LinearRegressionModel:
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
        path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(path + '/../', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        joblib.dump(self.model, folder_path + '/' + self.model_name + '_model.pkl')
        joblib.dump(self.scaler_X, folder_path + '/' + self.model_name + '_scalerX.pkl')
        joblib.dump(self.scaler_Y, folder_path + '/' + self.model_name + '_scalerY.pkl')
        print("Done!")

    def load_checkpoint(self, folder="checkpoints"):
        print("Loading checkpoint...")
        path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(path + '/../', folder)
        try:
            self.model = joblib.load(folder_path + '/' + self.model_name + '_model.pkl')
            self.scaler_X = joblib.load(folder_path + '/' + self.model_name + '_scalerX.pkl')
            self.scaler_Y = joblib.load(folder_path + '/' + self.model_name + '_scalerY.pkl')
            self.is_trained = True
            print("Done!")
        except FileNotFoundError:
            print("ERROR: no saved checkpoint found!")