import numpy as np
import os

def create_train_test_split(processed_data_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    print(f"Loading processed data from: {processed_data_path}")
    data = np.load(processed_data_path)
    
    print(f"Original dataset shape: {data.shape}")
    num_samples = data.shape[0]
    
    # Shuffle the data indices
    indices = np.random.permutation(num_samples)
    
    # Calculate split sizes
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)
    
    # Slice the data
    train_data = data[indices[:train_end]]
    val_data = data[indices[train_end:val_end]]
    test_data = data[indices[val_end:]]
    
    print(f"Train shape: {train_data.shape}")
    print(f"Validation shape: {val_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the splits
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    np.save(os.path.join(output_dir, 'val.npy'), val_data)
    np.save(os.path.join(output_dir, 'test.npy'), test_data)
    
    print(f"Splits successfully saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_FILE = r"D:\Work\CODE\Music Repo\data\processed\maestro_processed.npy"
    OUTPUT_DIR = r"D:\Work\CODE\Music Repo\data\train_test_split"
    
    create_train_test_split(INPUT_FILE, OUTPUT_DIR)