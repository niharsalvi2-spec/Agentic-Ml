import os
import torch
import numpy as np

class GenixDataLoader:
    """
    A highly optimized Data Loader for autoregressive model training.
    It reads from memory-mapped binary files (created by prepare.py), 
    ensuring we can train on terabytes of data without RAM overflow.
    """
    def __init__(self, data_dir: str, split: str, batch_size: int, block_size: int, device: str = 'cpu'):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        # Determine the binary file path
        bin_path = os.path.join(data_dir, f'{split}.bin')
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing {bin_path}. Run prepare.py first.")
            
        # We use numpy memmap to prevent loading the entire dataset into RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
    def get_batch(self):
        """
        Samples random chunks of length block_size from the dataset.
        Returns:
            x (Tensor): Input tokens
            y (Tensor): Target tokens (x shifted right by 1)
        """
        # Generate random starting indices for the batch
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        
        # Read the chunks from the memmap
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        # Pin memory for faster transfer to GPU if device is cuda
        if 'cuda' in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            
        return x, y
