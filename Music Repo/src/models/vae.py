import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64):
        """
        VAE implementation for Multi-Genre Music Generation[cite: 51, 96].
        
        Args:
            input_dim: Number of MIDI pitches (usually 88 for piano roll)[cite: 31].
            hidden_dim: Hidden units for LSTM layers.
            latent_dim: Dimension of the latent space (z)[cite: 57].
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # --- Encoder [cite: 145] ---
        # Using LSTM as suggested by Task 1 & 2 requirements [cite: 96, 111]
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder [cite: 151] ---
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Computes encoder outputs: (mu, sigma) [cite: 145]"""
        _, (h_n, _) = self.encoder_lstm(x)
        # Use the last hidden state to represent the sequence
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample latent vector using reparameterization: z = mu + sigma * epsilon [cite: 148]"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # epsilon ~ N(0, I)
        return mu + eps * std

    def decode(self, z, seq_len):
        """Decode reconstructed sequence [cite: 151]"""
        # Prepare hidden state from latent vector
        h_0 = self.decoder_fc(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        # Initial input for the decoder (e.g., zeros or a start token)
        decoder_input = torch.zeros(z.size(0), seq_len, self.decoder_lstm.input_size).to(z.device)
        
        output, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        reconstruction = torch.sigmoid(self.output_layer(output))
        return reconstruction

    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, seq_len), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Total VAE objective: L_vae = L_recon + beta * L_KL [cite: 160]
    """
    # Reconstruction loss: Binary Cross Entropy or MSE [cite: 154]
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: D_KL(q(z|X) || p(z)) [cite: 157]
    # Standard formula for Gaussian KL divergence [cite: 96]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss