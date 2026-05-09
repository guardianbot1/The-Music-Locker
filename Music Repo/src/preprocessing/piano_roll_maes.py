import os
from pathlib import Path
import pretty_midi
import numpy as np

def midi_to_pianoroll(midi_file, fs=16, seq_length=128):
    #fs=1 sec music divided into fs slices, seq_length=how many slices in 1 training example 

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file) # Load the MIDI file
              
        piano_roll = midi_data.get_piano_roll(fs=fs) # Extract piano roll: Returns shape (128 pitches, total_time_steps(128))        
        
        piano_roll = (piano_roll > 0).astype(np.float32) # Binarize the data: 1 if note is playing, 0 if not # This simplifies the task for a basic LSTM        
        
        piano_roll = piano_roll.T # Transpose.Now-->(total_time_steps(128), 128 pitches)
        
        # Segment into fixed-length windows
        sequences = []
        for i in range(0, piano_roll.shape[0] - seq_length, seq_length):
            segment = piano_roll[i : i + seq_length, :]       
            if segment.shape[0] == seq_length:
                sequences.append(segment)              
        return sequences
    
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return []

def process_all_midis(raw_dir, output_file, max_files=None):
    """
    Processes a batch of MIDI files and saves them as a single .npy file.
    """
    all_sequences = []
    
    # 1. SANITY CHECK: Does Python actually see the folder?
    print(f"Checking path: {raw_dir}")
    print(f"Does this folder exist? {os.path.exists(raw_dir)}")
    
    if not os.path.exists(raw_dir):
        print("ERROR: Python cannot find the directory! Check the RAW_DATA_PATH.")
        return

    # 2. THE FAIL-SAFE SEARCH: os.walk
    midi_files = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            # Check if it ends in .mid or .midi (ignoring uppercase/lowercase)
            if file.lower().endswith('.midi') or file.lower().endswith('.mid'):
                midi_files.append(os.path.join(root, file))
                
    print(f"Found {len(midi_files)} total MIDI files.")
    
    if len(midi_files) == 0:
        print("Still found 0 files. The folder exists, but no .midi files were seen by Python.")
        return

    # Process a subset to avoid running out of RAM initially
    for file in midi_files[:max_files]:
        print(f"Processing: {os.path.basename(file)}")
        seqs = midi_to_pianoroll(file)
        all_sequences.extend(seqs)
        
    # Convert list to a giant 3D numpy array: (Num_Samples, seq_length, 128)
    if len(all_sequences) > 0:
        final_dataset = np.array(all_sequences)
        print(f"\nFinal dataset shape: {final_dataset.shape}")
        
        # Save the processed data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, final_dataset)
        print(f"Dataset saved to {output_file}")
    else:
        print("\nFailed to extract any valid sequences from the files.")

if __name__ == "__main__":
    RAW_DATA_PATH = r"D:\Work\CODE\Music Repo\data\genres"
    OUTPUT_PATH = r"D:\Work\CODE\Music Repo\data\processed\maestro_processed.npy"
    
    process_all_midis(RAW_DATA_PATH, OUTPUT_PATH)