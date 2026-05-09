import torch
import numpy as np
import pretty_midi
import os
import sys

# Absolute path to your models folder so it can find autoencoder.py
MODELS_DIR = r"D:\Work\CODE\Music Repo\src\models"
sys.path.append(MODELS_DIR)
from autoencoder import LSTMAutoencoder

def matrix_to_midi(matrix, output_file, fs=16):
    """Converts a (128, 128) binary matrix back into a MIDI file."""
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    matrix = (matrix > 0.5).astype(int)

    for pitch in range(128):
        is_on = False
        start_time = 0
        for time_step in range(128):
            if matrix[time_step, pitch] == 1 and not is_on:
                start_time = time_step / fs
                is_on = True
            elif matrix[time_step, pitch] == 0 and is_on:
                end_time = time_step / fs
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
                piano.notes.append(note)
                is_on = False
        
        if is_on:
            end_time = 128 / fs
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_file)

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on device: {device}")
    
    # Absolute paths for your model and output directory
    model_path = r"D:\Work\CODE\Music Repo\outputs\lstm_autoencoder.pth"
    out_dir = r"D:\Work\CODE\Music Repo\outputs\generated_midis"
    os.makedirs(out_dir, exist_ok=True)
    
    model = LSTMAutoencoder(input_dim=128, hidden_dim=64, seq_len=128).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Successfully loaded trained model weights.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return

    print("Generating 5 MIDI samples...")
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, 64).to(device)
            repeated_vector = z.unsqueeze(1).repeat(1, 128, 1)
            decoder_out, _ = model.decoder(repeated_vector)
            out = model.fc(decoder_out)
            reconstructed = model.sigmoid(out)
            
            matrix = reconstructed.squeeze(0).cpu().numpy()
            
            out_file = os.path.join(out_dir, f"sample_{i+1}.mid")
            matrix_to_midi(matrix, out_file, fs=16)
            print(f"Saved: {out_file}")

if __name__ == "__main__":
    generate()