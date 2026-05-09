import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# Add the models directory to the path so we can import the autoencoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from autoencoder import LSTMAutoencoder

class MidiDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

def train_model():
    data_dir = r"D:\Work\CODE\Music Repo\data\train_test_split\maestro_splits"
    train_path = os.path.join(data_dir, 'train.npy')
    val_path = os.path.join(data_dir, 'val.npy')
    
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_dataset = MidiDataset(train_path)
    val_dataset = MidiDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64, seq_len=128).to(device)
    
    # UPDATED: Per the project guidelines, using Mean Squared Error (MSE)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # To store losses for the required deliverable curve
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item() * batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # Save model and loss history
    os.makedirs(os.path.join("..", "..", "outputs"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join("..", "..", "outputs", "lstm_autoencoder.pth"))
    np.save(os.path.join("..", "..", "outputs", "loss_history_ae.npy"), history)
    print("Training complete! Model and Loss History saved.")

if __name__ == "__main__":
    train_model()