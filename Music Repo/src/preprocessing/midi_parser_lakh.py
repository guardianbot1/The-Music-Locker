from pathlib import Path
import torch
import os

from piano_roll_lakh import midi_to_piano_roll

class LakhMidiDataset:
    def __init__(self, root_dir, seq_len=128, max_files=5000):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.data = []
        
        print(f"Searching for MIDI files in {self.root_dir}...")
        all_midi_files = list(self.root_dir.rglob("*.mid")) + list(self.root_dir.rglob("*.midi"))
        print(f"Found {len(all_midi_files)} files. Processing up to {max_files} files to save RAM...")
        
        files_processed = 0
        for midi_path in all_midi_files:
            if files_processed >= max_files:
                break
                
            segments = midi_to_piano_roll(str(midi_path), fs=16, seq_len=self.seq_len)
            if segments is not None:
                self.data.append(torch.FloatTensor(segments))
                files_processed += 1
                
        if len(self.data) > 0:
            self.data = torch.cat(self.data, dim=0)
            print(f"Dataset ready! Total sequence windows: {self.data.shape[0]}")
        else:
            print("Warning: No valid MIDI files were processed.")

if __name__ == "__main__":
    RAW_DATA_DIR = r"D:\Work\CODE\Music Repo\data\raw_midi\lakh" 
    PROCESSED_DATA_PATH = r"D:\Work\CODE\Music Repo\data\processed\lakh_matrix.pt"
    
    print("Starting dataset preprocessing...")
    
    # We pass max_files=5000 here to prevent the 'Killed' crash
    dataset = LakhMidiDataset(RAW_DATA_DIR, seq_len=128, max_files=5000)
    matrix = dataset.data
    
    if len(matrix) > 0:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        torch.save(matrix, PROCESSED_DATA_PATH)
        print(f"Success! Matrix of shape {matrix.shape} saved to {PROCESSED_DATA_PATH}")