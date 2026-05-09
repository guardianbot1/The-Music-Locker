import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# --- NEW: Fix for ModuleNotFoundError ---
# This adds "D:\Work\CODE\Music Repo" to your Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------------------------------

# Now this import will work!
from src.models.vae import VAE, vae_loss_function


def train():
    # 1. Direct Paths
    TRAIN_DATA_PATH = r"D:\Work\CODE\Music Repo\data\train_test_split\lakh_train.pt"
    VAL_DATA_PATH = r"D:\Work\CODE\Music Repo\data\train_test_split\lakh_val.pt"
    MODEL_SAVE_PATH = r"D:\Work\CODE\Music Repo\outputs\vae_model.pth"
    HISTORY_SAVE_PATH = r"D:\Work\CODE\Music Repo\outputs\vae_loss_history.pt"

    # 2. Hyperparameters (Task 2 Specifications)
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    BETA = 1.0  # Weight for KL Divergence as per Algorithm 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================Using device: {device}")
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    

    # 3. Load Datasets
    print("Loading split datasets...")
    train_tensor = torch.load(TRAIN_DATA_PATH)
    val_tensor = torch.load(VAL_DATA_PATH)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE)

    # 4. Initialize Model
    model = VAE(input_dim=88, hidden_dim=256, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dictionary to track metrics for the report deliverables
    history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_total_loss': []
    }

    # 5. Training Loop (Algorithm 2)
    print("Starting VAE Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass: Encode -> Sample (Reparameterize) -> Decode
            recon_batch, mu, logvar = model(data)
            
            # Mathematical Formulation [cite: 59, 160]
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + (BETA * kl_loss)
            
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        # Validation Step
        model.eval()
        val_total = 0
        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                v_loss = vae_loss_function(recon, data, mu, logvar, beta=BETA)
                val_total += v_loss.item()

        # Save metrics to history
        num_train = len(train_loader.dataset)
        history['train_total_loss'].append(train_loss / num_train)
        history['train_recon_loss'].append(train_recon / num_train)
        history['train_kl_loss'].append(train_kl / num_train)
        history['val_total_loss'].append(val_total / len(val_loader.dataset))

        print(f"Epoch {epoch}: Loss: {history['train_total_loss'][-1]:.4f} "
              f"(Recon: {history['train_recon_loss'][-1]:.4f}, KL: {history['train_kl_loss'][-1]:.4f})")

        # Save Model State
        if epoch % 10 == 0:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 6. Final Save for Deliverables [cite: 60, 61]
    os.makedirs(os.path.dirname(HISTORY_SAVE_PATH), exist_ok=True)
    torch.save(history, HISTORY_SAVE_PATH)
    print(f"\nSuccess! Training complete.")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Loss history saved to: {HISTORY_SAVE_PATH}")

if __name__ == "__main__":
    train()