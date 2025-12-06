import wandb

from wandb.integration.keras import WandbMetricsLogger
from src.fnn import FNNModel

def train(fnn: FNNModel, train_x, train_y, config=None, new_train=False):
    config = config if config is not None else {"lr": 0.001,
                                                "epochs": 50,
                                                "batch_size": 32,
                                                "hidden_layers": [64, 64],
                                                "input_dim": train_x.shape[0],
                                                "output_dim": train_y.shape[0]}

    fnn.build_model(config.hidden_layers, config.lr)
    """
    [WARNING] PROBABLY OUTDATED SINCE THE REST OF THE PROJECT HAS MOVED TO WandB 

    CODE TO HANDLE CHECKPOINTS 
    # check if we have to resume training from an old checkpoint/history
    path = os.path.dirname(os.path.abspath(__file__))
    if glob(path + '/../checkpoints/' + fnn.model_name + '.keras') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_scalerX.pkl') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_scalerY.pkl') and \
        glob(path + '/../checkpoints/' + fnn.model_name + '_historyHistory.pkl') and \
        not new_train:
        print("Loading checkpoint...")
        fnn.load_checkpoint()
        print("Done!")
    """
    print("Training...")
    # TODO: forse loop sugli iperparametri della FNN
    fnn.train(train_x, train_y, config.epochs, callback=[WandbMetricsLogger()])
    print("Done!")


def evaluate(fnn: FNNModel, test_x, test_y):
    fnn.evaluate(test_x, test_y)