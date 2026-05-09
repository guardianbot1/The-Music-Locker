import numpy as np
import pretty_midi

def midi_to_piano_roll(midi_path, fs=16, seq_len=128):
    """
    Converts a single MIDI file to a piano roll matrix.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # Keep only standard 88 keys (MIDI pitches 21 to 108)
        piano_roll = piano_roll[21:109, :]
        piano_roll = piano_roll.T # Transpose to (time_steps, 88)
        
        # Normalize velocities
        piano_roll = piano_roll / 127.0
        
        # Segment into fixed-length windows
        num_windows = piano_roll.shape[0] // seq_len
        if num_windows == 0:
            return None
            
        piano_roll = piano_roll[:num_windows * seq_len, :]
        segments = piano_roll.reshape(num_windows, seq_len, 88)
        return segments
        
    except Exception:
        # Ignore corrupted or empty MIDI files
        return None