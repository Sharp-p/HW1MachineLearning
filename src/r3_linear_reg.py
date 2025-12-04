from glob import glob
from linear_regression import *

import pandas as pd
import os

def train(train_x, train_y, model):
    # preprocess data (normalization) and training
    path = os.path.dirname(os.path.abspath(__file__))
    if (glob(path + '/../checkpoints/' + model.model_name + '_model.pkl') and
        glob(path + '/../checkpoints/' + model.model_name + '_scalerX.pkl') and
        glob(path + '/../checkpoints/' + model.model_name + '_scalerY.pkl')):
        # since the training is a mathematical operation more training does not exist
        # so we only upload the checkpoint
        model.load_checkpoint()
    else:
        model.train(train_x, train_y)
        model.save_checkpoint()

def eval(test_x, test_y, model):
    # linear regression
    model.evaluate(test_x, test_y)


def load_data():
    path = os.path.dirname(os.path.abspath(__file__))
    train_ds = pd.read_csv(path + '/../datasets/reacher3_train_1.csv')
    input_training = train_ds.iloc[:, :7].to_numpy()
    output_training = train_ds.iloc[:, -3:].to_numpy()

    test_ds = pd.read_csv(path + '/../datasets/reacher3_test_1.csv')
    input_test = test_ds.iloc[:, :7].to_numpy()
    output_test = test_ds.iloc[:, -3:].to_numpy()
    return input_training, output_training, input_test, output_test

def train_eval() -> LinearRegressionModel:
    train_x, train_y, test_x, test_y = load_data()
    model = LinearRegressionModel(model_name= "reacher3_linear")
    train(train_x, train_y, model)
    eval(test_x, test_y, model)

    return model

if __name__ == "__main__":
    train_eval()