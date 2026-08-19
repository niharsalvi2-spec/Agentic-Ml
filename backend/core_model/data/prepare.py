import os
import urllib.request
import numpy as np
from tokenizer import GenixTokenizer

def prepare_tinyshakespeare():
    """
    Downloads the TinyShakespeare dataset and processes it into memory-mapped
    numpy arrays for extremely efficient GPU/CPU ingestion during training.
    """
    data_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(data_dir, 'input.txt')

    # 1. Download data if it doesn't exist
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading dataset from {data_url}...")
        urllib.request.urlretrieve(data_url, input_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Dataset length: {len(data):,} characters")

    # 2. Split into train and validation sets (90/10 split)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # 3. Tokenize
    tokenizer = GenixTokenizer()
    print("Encoding train split...")
    train_ids = tokenizer.encode(train_data)
    print("Encoding validation split...")
    val_ids = tokenizer.encode(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # 4. Save to binary files (uint16 is enough for vocab size up to 65k)
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_bin_path = os.path.join(data_dir, 'train.bin')
    val_bin_path = os.path.join(data_dir, 'val.bin')
    
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)
    
    print(f"Saved memory-mappable binaries to {train_bin_path} and {val_bin_path}")

if __name__ == '__main__':
    prepare_tinyshakespeare()
