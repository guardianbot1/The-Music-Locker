import torch
import os

def split_dataset():
    # 1. Define your exact direct paths
    INPUT_PATH = r"D:\Work\CODE\Music Repo\data\processed\lakh_matrix.pt"
    OUTPUT_DIR = r"D:\Work\CODE\Music Repo\data\train_test_split"
    
    print(f"Loading matrix from {INPUT_PATH}...")
    # Load the massive matrix we just made
    data = torch.load(INPUT_PATH)
    total_sequences = data.shape[0]
    print(f"Successfully loaded {total_sequences} sequences.")

    # 2. Shuffle the data
    # This is crucial so we don't have all the songs from one genre clumped together
    print("Shuffling data...")
    indices = torch.randperm(total_sequences)
    data = data[indices]

    # 3. Calculate the split sizes (80% Train, 10% Val, 10% Test)
    train_size = int(0.8 * total_sequences)
    val_size = int(0.1 * total_sequences)
    
    # 4. Slice the tensor into three chunks
    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    print("\nSplit Results:")
    print(f"Train Set: {train_data.shape}")
    print(f"Val Set:   {val_data.shape}")
    print(f"Test Set:  {test_data.shape}")

    # 5. Save the new splits to disk
    print("\nSaving splits to disk...")
    torch.save(train_data, os.path.join(OUTPUT_DIR, "lakh_train.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "lakh_val.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "lakh_test.pt"))
    
    print("Success! Data splitting is complete.")

if __name__ == "__main__":
    split_dataset()