import torch
import numpy as np
import os
import sys

# --- Path Fix ---
project_root = r"D:\Work\CODE\Music Repo"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.vae import VAE
from midi_export import piano_roll_to_midi

def interpolate_latent_space(num_steps=8):
    # 1. Setup paths and parameters
    MODEL_PATH = r"D:\Work\CODE\Music Repo\outputs\lakh\vae_model.pth"
    OUTPUT_DIR = r"D:\Work\CODE\Music Repo\outputs\generated_midis\interpolation"
    LATENT_DIM = 64
    SEQ_LEN = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load the trained VAE
    model = VAE(input_dim=88, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print(f"Performing interpolation experiment with {num_steps} steps...")

    with torch.no_grad():
        # 3. Pick two random points (z_start and z_end) from N(0, I)
        z_start = torch.randn(1, LATENT_DIM).to(device)
        z_end = torch.randn(1, LATENT_DIM).to(device)

        # 4. Interpolate between them: z_i = (1 - alpha) * z_start + alpha * z_end
        alphas = np.linspace(0, 1, num_steps)
        for i, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z_start + alpha * z_end
            
            # Decode the interpolated point
            recon_roll = model.decode(z_interp, SEQ_LEN)
            roll = recon_roll[0].cpu().numpy()
            
            # 5. Export to MIDI
            pm = piano_roll_to_midi(roll)
            filename = os.path.join(OUTPUT_DIR, f"interp_step_{i+1}_alpha_{alpha:.2f}.mid")
            pm.write(filename)
            print(f"Saved step {i+1}/{num_steps}: {filename}")

if __name__ == "__main__":
    interpolate_latent_space()