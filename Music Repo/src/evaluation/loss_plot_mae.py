import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_curve():
    # Absolute paths for your history file and plots directory
    history_path = r"D:\Work\CODE\Music Repo\outputs\loss_history_ae.npy"
    plot_dir = r"D:\Work\CODE\Music Repo\outputs\plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        history = np.load(history_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Loss history not found at {history_path}.")
        return

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training MSE Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Validation MSE Loss', linewidth=2)
    
    plt.title('Task 1: LSTM Autoencoder Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    
    plot_file = os.path.join(plot_dir, 'reconstruction_loss_curve.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Successfully saved loss curve to: {plot_file}")
    plt.show()

if __name__ == "__main__":
    plot_loss_curve()