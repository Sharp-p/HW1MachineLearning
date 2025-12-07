import itertools
import os

import pandas as pd
import wandb
from keras.src.callbacks import EarlyStopping

from wandb.integration.keras import WandbMetricsLogger
from src.fnn import FNNModel

def train(fnn: FNNModel, train_x, train_y, config=None, new_train=False):
    config = config if config is not None else {"lr": 0.001,
                                                "epochs": 50,
                                                "batch_size": 32,
                                                "hidden_layers": [64, 64],
                                                "val_split": 0.2,
                                                "activation": "relu",
                                                "input_dim": train_x.shape[0],
                                                "output_dim": train_y.shape[0]}

    fnn.build_model(config["hidden_layers"], config["lr"], config["activation"])
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
    fnn.train(train_x, train_y,
              config["epochs"],
              config["batch_size"],
              config["val_split"],
              callback=[WandbMetricsLogger(), EarlyStopping(patience=20,
                                                            restore_best_weights=True)],)
    print("Done!")

def evaluate(fnn: FNNModel, test_x, test_y, config):
    pred, mse, r2 = fnn.evaluate(test_x, test_y)
    result_data = config.copy()
    result_data["mse"] = mse
    result_data["r2"] = r2

    df_result = pd.DataFrame([result_data])

    file_name = "exp_summary.csv"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log")
    # check the existence of folder
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, file_name)
    # check the existence of file
    if not os.path.exists(path):
        df_result.to_csv(path, index=False, mode='w')
    else:
        df_result.to_csv(path, index=False, mode='a', header=False)

    print(f"Saved scores for {fnn.model_name}!")


def train_eval(train_x, train_y, test_x, test_y, PROJECT_NAME):
    # calculates which dataset is this
    N = train_y.shape[1]
    # creates the configs for this series of experiments
    param_grid = {"lr": [0.1, 0.01, 0.001, 0.0001],
                  "batch_size": [16, 32, 64, 128],
                  "hidden_layers": [[1280],
                                    [640, 640],
                                    [427, 427, 427],
                                    [320, 320, 320, 320],
                                    [256, 256, 256, 256, 256],
                                    [213, 213, 213, 213, 213, 213],
                                    [183, 183, 183, 183, 183, 183, 183],
                                    [160, 160, 160, 160, 160, 160, 160, 160],
                                    [142, 142, 142, 142, 142, 142, 142, 142, 142],
                                    [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
                  "activation": ["relu", "tanh"],
                  "val_split": [0.125],
                  "epochs": [400],
                  "dataset": [f"Reacher{N}"]}

    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # loops over every configuration for this fnn
    for i, config in enumerate(configs):
        # start an experiment with a configuration
        with wandb.init(project=PROJECT_NAME,
                        entity="sharp-1986413-sapienza-universit-di-roma",
                         config=config,
                         group=config["dataset"],
                         job_type="train",
                         name=f"{config['dataset']}_arch{len(config['hidden_layers'])}_lr{config['lr']}_batch{config['batch_size']}_valS{config['val_split']}_{config['activation']}"
                        ) as run:
            print("============================================================")
            print(f"TRAINING CONFIG {i}: {run.name}...")
            print("============================================================")
            # creating a model for this configuration and training it
            fnn = FNNModel(input_dim=train_x.shape[1],
                           output_dim=train_y.shape[1],
                           model_name=run.name)
            # doing the actual training
            train(fnn, train_x, train_y, config)
            evaluate(fnn, test_x, test_y, config)
            # end this experiment
            fnn.save_checkpoint()
            print("Done!")

