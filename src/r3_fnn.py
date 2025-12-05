import os
from glob import glob

from src.fnn import FNNModel
from commons.data import load_data

def train(fnn: FNNModel, train_x, train_y, new_train = False):
    fnn.build_model()
    path = os.path.dirname(os.path.abspath(__file__))
    if glob(path + '/../checkpoints/' + fnn.model_name + '.keras') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_scalerX.pkl') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_scalerY.pkl') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_historyHistory.pkl') and \
        not new_train:
        print("Loading checkpoint...")
        fnn.load_checkpoint()
        print("Done!")

    print("Training...")
    # TODO: forse loop sugli iperparametri della FNN
    fnn.train(train_x, train_y)
    print("Done!")

def evaluate(fnn: FNNModel, test_x, test_y):
    fnn.evaluate(test_x, test_y)

def train_eval():
    train_x, train_y, test_x, test_y = load_data()
    fnn = FNNModel(input_dim=7, output_dim=3, model_name="reacher3_fnn")
    train(fnn, train_x, train_y)
    evaluate(fnn, test_x, test_y)
    fnn.plot_loss()
    fnn.save_checkpoint()

if __name__ == "__main__":
    train_eval()