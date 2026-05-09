import numpy as np
import pretty_midi
import os

def piano_roll_to_midi(piano_roll, fs=16, program=0):
    """
    Converts a 2D piano roll (time x pitch) into a PrettyMIDI object.
    """
    # Parameters: piano_roll shape is (seq_len, 88)
    # We need to pad it to 128 pitches for standard MIDI
    padded_roll = np.zeros((piano_roll.shape[0], 128))
    # Map our 88 notes back to MIDI pitches 21-108
    padded_roll[:, 21:109] = piano_roll 
    
    # Threshold the sigmoid outputs to get binary note-on/off
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    # Use a simple threshold (e.g., 0.5) to decide if a note is active
    velocity_roll = (padded_roll > 0.5) * 100 
    
    # For every pitch, find contiguous blocks of 'on' notes
    for pitch in range(128):
        for start, end in _find_contiguous_blocks(velocity_roll[:, pitch]):
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start / fs,
                end=end / fs
            )
            instrument.notes.append(note)
            
    pm.instruments.append(instrument)
    return pm

def _find_contiguous_blocks(arr):
    """Finds start and end indices of contiguous True values in a boolean array."""
    blocks = []
    start = None
    for i, val in enumerate(arr):
        if val and start is None:
            start = i
        elif not val and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(arr)))
    return blocks