from commons.data import load_data
from commons.experimenter import train_eval

def main():
    # for every dataset
    for N in [6]:
        # loading the correct dataset
        train_x, train_y, test_x, test_y = load_data(N)
        train_eval(train_x, train_y, test_x, test_y, "HW1ML_rN")

if __name__ == "__main__":
    main()