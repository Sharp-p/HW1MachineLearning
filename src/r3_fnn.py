from src.fnn import FNNModel

def train(fnn: FNNModel):
    fnn.build_model()
    fnn.train()

def train_eval():

    fnn = FNNModel(input_dim=7, output_dim=3, model_name="reacher3_fnn")
    train()
    evaluate()

if __name__ == "__main__":
    train_eval()