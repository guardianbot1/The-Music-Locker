from pathlib import Path
import shutil

# Source: MAESTRO dataset folder
SOURCE_DIR = Path(r"D:\Work\CODE\Music Repo\data\raw_midi\maestro-v3.0.0")

# Destination: classical genre folder
DEST_DIR = Path(r"D:\Work\CODE\Music Repo\data\raw_midi\genre_sorted\classical")

# Create destination folder if it doesn't exist
DEST_DIR.mkdir(parents=True, exist_ok=True)

# MIDI extensions to copy
MIDI_EXTENSIONS = {".mid", ".midi"}

copied = 0
skipped = 0

for midi_file in SOURCE_DIR.rglob("*"):
    if midi_file.is_file() and midi_file.suffix.lower() in MIDI_EXTENSIONS:
        destination_file = DEST_DIR / midi_file.name

        # Avoid overwriting files with same name
        if destination_file.exists():
            destination_file = DEST_DIR / f"{midi_file.stem}_{copied}{midi_file.suffix}"

        shutil.copy2(midi_file, destination_file)
        copied += 1
        print(f"Copied: {midi_file.name}")

print("\nDone.")
print(f"Copied MIDI files: {copied}")
print(f"Destination: {DEST_DIR}")