import torch
import matplotlib.pyplot as plt
import os

def plot_vae_metrics():
    # 1. Define paths
    HISTORY_PATH = r"D:\Work\CODE\Music Repo\outputs\lakh\vae_loss_history.pt"
    SAVE_DIR = r"D:\Work\CODE\Music Repo\outputs\plots"
    
    if not os.path.exists(HISTORY_PATH):
        print(f"Error: Could not find history file at {HISTORY_PATH}")
        return

    # 2. Load the recorded history
    history = torch.load(HISTORY_PATH)
    epochs = range(1, len(history['train_recon_loss']) + 1)

    # 3. Create Plot 1: Reconstruction Loss Curve (Task 2 Requirement)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_recon_loss'], 'b-', label='Training Reconstruction Loss')
    plt.title('Task 2: VAE Reconstruction Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "reconstruction_loss.png"))
    print("Saved reconstruction_loss.png")

    # 4. Create Plot 2: KL-Divergence Loss (Task 2 Requirement)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_kl_loss'], 'r-', label='KL-Divergence')
    plt.title('Task 2: Latent Space Regularization (KL Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('KL Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "kl_divergence_loss.png"))
    print("Saved kl_divergence_loss.png")

    # 5. Create Plot 3: Combined Total Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_total_loss'], 'g-', label='Total VAE Loss')
    plt.title('Task 2: Total Objective Function (Recon + KL)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "total_vae_loss.png"))
    print("Saved total_vae_loss.png")

    plt.show()

if __name__ == "__main__":
    plot_vae_metrics()