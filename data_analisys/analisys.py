import os

import numpy as np
import pandas as pd

from commons.data import load_data
from src.fnn import FNNModel


def linear_samp_pred_csv(Y_test, y_pred):
    actual_values = Y_test[:, 0]
    predicted_values = y_pred[:, 0]

    comparison_df = pd.DataFrame({
        'Actual': actual_values,
        'Predicted': predicted_values
    })

    # Take random 100 points
    n_samples = min(100, len(comparison_df))
    sample_df = comparison_df.sample(n=n_samples, random_state=42)

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "..", "log", "linear_vs_actual_sample.csv")

    # Save the file in log
    sample_df.to_csv(path, index=False, sep=' ')


def correlations_test():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Load the dataset
    df = pd.read_csv('../log/exp_summary.csv')

    # 1. Check the Best Results per Robot (to check for the "Best R2")
    print("--- Best Results per Robot ---")
    best_indices = df.groupby('dataset')['r2'].idxmax()
    print(df.loc[best_indices][['dataset', 'r2', 'mse', 'hidden_layers', 'lr', 'batch_size', 'activation']])

    # 2. Tanh vs LR Stability
    # We look at the average R2 and the minimum R2 (to spot failures) for Tanh vs Relu across LRs
    print("\n--- Stability Analysis: Activation vs LR (Mean R2) ---")
    stability = df.groupby(['activation', 'lr'])['r2'].mean().unstack()
    print(stability)

    print("\n--- Stability Analysis: Activation vs LR (Min R2 - to check for crashes) ---")
    stability_min = df.groupby(['activation', 'lr'])['r2'].min().unstack()
    print(stability_min)

    # 3. Batch Size Impact
    # We look at the mean R2 for each batch size, split by Robot
    print("\n--- Batch Size Impact: Mean R2 per Dataset ---")
    batch_impact = df.groupby(['dataset', 'batch_size'])['r2'].mean().unstack()
    print(batch_impact)

    # Check top performers distribution by batch size
    print("\n--- Batch Size of Top 10 Models per Dataset ---")
    for robot in df['dataset'].unique():
        print(f"\n{robot} Top 5:")
        print(df[df['dataset'] == robot].nlargest(5, 'r2')[['r2', 'batch_size', 'lr', 'activation']])

def generate_prediction_csv(X_test, Y_test, model_name="reacher3_best_fnn", input_dim=7, output_dim=3):
    """
    Carica il modello, fa predizioni e salva un CSV per i grafici LaTeX.
    """
    # 1. Istanziazione del modello
    fnn = FNNModel(input_dim=input_dim, output_dim=output_dim, model_name=model_name)

    # 2. Caricamento Checkpoint
    try:
        fnn.load_checkpoint(folder="checkpoints")
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return

    print("Avvio predizione sul Test Set...")
    y_pred, mse, r2 = fnn.evaluate(X_test, Y_test)

    # Campioniamo 150 punti
    data = {}

    # Aggiungiamo colonne per ogni giunto (q1, q2, q3)
    for i in range(output_dim):
        data[f'Actual_q{i + 1}'] = Y_test[:, i]
        data[f'Pred_q{i + 1}'] = y_pred[:, i]

    df = pd.DataFrame(data)

    # Random sampling
    df_sample = df.sample(n=150, random_state=42)

    # Saving

    path = os.path.dirname(os.path.abspath(__file__))
    filename = f'{model_name}_predictions_sample.csv'
    path = os.path.join(path, "..", "log", filename)
    df_sample.to_csv(path, index=False, sep=' ')
    print(f"File salvato con successo: {filename}")
    print(f"Performance rilevate -> MSE: {mse:.5f}, R2: {r2:.4f}")

if __name__ == "__main__":
    # generating the prediction in csv for plotting with the best reacher3 model
    train_x, train_y, test_x, test_y = load_data(3)
    generate_prediction_csv(train_x, train_y, "Reacher3_arch2_lr0.001_batch16_valS0.05_relu")

    #correlations_test()
