import numpy as np
import matplotlib.pyplot as plt

# 1. Load your processed data
data_path = r"D:\Work\CODE\Music Repo\data\processed\maestro_processed.npy"
data = np.load(data_path)

# 2. Pick a random sample (e.g., the 5th musical phrase)
sample_idx = 5
sample = data[sample_idx]  # This is a (128, 128) matrix

print(f"Visualizing Sample #{sample_idx}")
print(f"Shape: {sample.shape}")

# 3. Plot it
plt.figure(figsize=(10, 10))
# 'aspect=auto' makes the grid square-ish
# 'origin=lower' puts the low notes at the bottom, high notes at the top
plt.imshow(sample.T, aspect='auto', cmap='gray', origin='lower')

plt.title(f"Piano Roll Visualization (Sample {sample_idx})")
plt.xlabel("Time Steps (Forward ->)")
plt.ylabel("Pitch (Low -> High)")
plt.colorbar(label="Note On (1) / Off (0)")

plt.show()