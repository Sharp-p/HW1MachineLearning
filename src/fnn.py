import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.python.keras.initializers.initializers_v2 import Initializer


class FNNModel:
    def __init__(self, input_dim, output_dim, model_name="reacherN_fnn"):
        """
        Initializer
        """
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.model = None
        self.history = None
        self.is_trained = False

    def build_model(self, hidden_layers=[64, 64], learning_rate=0.001):
        print("Building model...")
        # sequential model theoretically not ideal for multiple input/outputs
        model = Sequential()

        # input layer
        model.add(Input(shape=(self.input_dim,)))

        # hidden layers (with ReLU activation function for non-linearity)
        for neurons in hidden_layers:
            model.add(Dense(neurons, activation="relu"))

        # output layer
        model.add(Dense(self.output_dim, activation="linear"))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        self.model = model

        print("Model ", self.model, " built!")

    def train(self, X_train, Y_train, epochs=50, batch_size=32, val_split=0.2) -> None:
        """
        Normalize data and train network

        :param X_train: the input data to train on
        :param Y_train: the expected output data to train on
        :param epochs: epochs to train the network
        :param batch_size: size of batches
        :param val_split: split size of validation set
        """

        if self.model is None:
            self.build_model()

        print("Data normalization...")
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)
        print("Done!")

        print("Training network for ", epochs, "epochs...")
        self.history = self.model.fit(
            X_scaled, Y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1
        )
        self.is_trained = True
        print("Done!")

    def evaluate(self, X_test, Y_test) -> float:
        """
        Evaluate network on normalized datas
        :param X_test: test data
        :param Y_test: expected output data
        """
        if not self.is_trained:
            print("Not trained!")
            return None

        X_test_scaled = self.scaler_X.transform(X_test)


        # prediction
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_Y.inverse_transform(y_pred_scaled)

        # evaluation
        mse = mean_squared_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)

        print("MSE: ", mse)
        print("R2 score: ", r2)
        return y_pred

    def plot_loss(self):
        """
        Visualize loss curve
        """
        if self.history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.history['loss'], label="Train Loss")
            plt.plot(self.history.history['val_loss'], label="Validation Loss")
            plt.title(f'Loss curve - {self.model_name}')
            plt.ylabel('MSE (Scaled)')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.show()

    def save_checkpoint(self, folder="checkpoints"):
        """
        Save the Keras model and the scikit-learn scalers
        :param folder: folder to save the model and scikit-learn scalers,
            relative to the project root
        """
        path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(path + '/../', folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Folder created!")

        print("Saving model checkpoint and data scalers...")
        # saving to keras's native format
        self.model.save(folder_path + '/' + self.model_name + '.h5')

        # saving scalers
        joblib.dump(self.scaler_X, folder_path + '/' + self.model_name + '.scalerX.pkl')
        joblib.dump(self.scaler_Y, folder_path + '/' + self.model_name + '.scalerY.pkl')
        print("Done!")

    def load_checkpoint(self, folder="checkpoints"):
        """
        Loading model and scaler.
        :param folder: the folder to load the model from,
            relative to the project root
        :return:
        """
        path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(path + '/../', folder)
        try:
            self.model = load_model(folder_path + '/' + self.model_name + '.h5')
            self.scaler_X = joblib.load(folder_path + '/' + self.model_name + '.scalerX.pkl')
            self.scaler_Y = joblib.load(folder_path + '/' + self.model_name + '.scalerY.pkl')
            print("Checkpoint and data scalers loaded!")
        except FileNotFoundError:
            print("Checkpoint and data scalers not found!")
