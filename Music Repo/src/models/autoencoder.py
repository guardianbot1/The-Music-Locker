import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, seq_len=128):
        """
        input_dim: 128 (The number of possible MIDI pitches)
        hidden_dim: The size of the compressed 'bottleneck' (e.g., 64)
        seq_len: 128 (The number of time steps / 8 bars)
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # --- ENCODER ---
        # Takes the input sequence and compresses it.
        # batch_first=True means we expect input shape: (Batch, Sequence, Features)
        self.encoder = nn.LSTM(input_size=input_dim, 
                               hidden_size=hidden_dim, 
                               batch_first=True)
        
        # --- DECODER ---
        # Takes the compressed representation and reconstructs the sequence.
        self.decoder = nn.LSTM(input_size=hidden_dim, 
                               hidden_size=hidden_dim, 
                               batch_first=True)
        
        # The output layer maps the hidden state back to the 128 pitches
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Sigmoid squashes the output values between 0 and 1 (since our notes are binary)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch_Size, 128, 128)
        
        # 1. ENCODER PASS
        # We only care about the final hidden state of the encoder
        _, (hidden, _) = self.encoder(x)
        
        # Extract the hidden state from the last layer 
        # latent_vector shape: (Batch_Size, hidden_dim)
        latent_vector = hidden[-1] 
        
        # 2. PREPARE DECODER INPUT
        # To generate 128 time steps, we repeat the latent vector 128 times
        # repeated_vector shape: (Batch_Size, 128, hidden_dim)
        repeated_vector = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 3. DECODER PASS
        # decoder_out shape: (Batch_Size, 128, hidden_dim)
        decoder_out, _ = self.decoder(repeated_vector)
        
        # 4. FINAL OUTPUT
        # Map back to 128 pitches: (Batch_Size, 128, 128)
        out = self.fc(decoder_out)
        reconstructed = self.sigmoid(out)
        
        return reconstructed

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing LSTM Autoencoder architecture...")
    
    # Create a dummy batch of 5 musical phrases (5, 128, 128)
    dummy_input = torch.rand(5, 128, 128)
    
    # Initialize the model
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64, seq_len=128)
    
    # Pass the dummy data through the model
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("\nSuccess! The model successfully reconstructed the input shape.")
    else:
        print("\nError: The output shape doesn't match the input.")