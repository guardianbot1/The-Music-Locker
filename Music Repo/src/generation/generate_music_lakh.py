import torch
import os

import sys

# --- Path Fix for ModuleNotFoundError ---
project_root = r"D:\Work\CODE\Music Repo"
if project_root not in sys.path:
    sys.path.append(project_root)

    
from src.models.vae import VAE
from midi_export import piano_roll_to_midi

def generate_samples(num_samples=8):
    # 1. Configuration
    MODEL_PATH = r"D:\Work\CODE\Music Repo\outputs\lakh\vae_model.pth"
    OUTPUT_DIR = r"D:\Work\CODE\Music Repo\outputs\generated_midis"
    LATENT_DIM = 64
    SEQ_LEN = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model
    model = VAE(input_dim=88, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Generating {num_samples} samples from latent space...")

    with torch.no_grad():
        # 3. Sample z ~ N(0, I) as per Algorithm 2
        z = torch.randn(num_samples, LATENT_DIM).to(device)
        
        # 4. Decode z into piano-roll sequences
        generated_rolls = model.decode(z, SEQ_LEN) # Shape: (8, 128, 88)
        
    # 5. Export to MIDI
    for i in range(num_samples):
        roll = generated_rolls[i].cpu().numpy()
        pm = piano_roll_to_midi(roll)
        
        filename = os.path.join(OUTPUT_DIR, f"vae_sample_{i+1}.mid")
        pm.write(filename)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    generate_samples()