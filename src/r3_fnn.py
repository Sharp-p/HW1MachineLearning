import os
from glob import glob

from src.fnn import FNNModel
from commons.data import load_data
from commons.experimenter import train, evaluate

def train_eval():
    # TODO: generare esperimenti per r3
    train_x, train_y, test_x, test_y = load_data()

    # TODO: loop sull'istanzazione della fnn e il training con esperimenti diversi
    fnn = FNNModel(input_dim=7, output_dim=3, model_name="reacher3_fnn")
    train(fnn, train_x, train_y)
    evaluate(fnn, test_x, test_y)
    fnn.plot_loss()
    # not using checkpoints since WandB implementations, integration not yet considered
    #fnn.save_checkpoint()

if __name__ == "__main__":
    train_eval()