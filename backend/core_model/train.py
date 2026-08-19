import os
import torch
import time
from data.dataloader import GenixDataLoader
from genix_model import GenixModel
from config import GenixConfig

def train():
    """
    Main training loop for Genix.
    """
    # -----------------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------------
    batch_size = 12        # How many independent sequences will we process in parallel?
    block_size = 64        # What is the maximum context length for predictions?
    max_iters = 100        # How many training iterations
    eval_interval = 20     # How often to evaluate the model on the validation set
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing Genix Training on device: {device}")
    
    # -----------------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------------
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    try:
        train_loader = GenixDataLoader(data_dir, split='train', batch_size=batch_size, block_size=block_size, device=device)
        val_loader = GenixDataLoader(data_dir, split='val', batch_size=batch_size, block_size=block_size, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python data/prepare.py' first.")
        return

    # -----------------------------------------------------------------------------
    # Initialize Model
    # -----------------------------------------------------------------------------
    config = GenixConfig.get_nano_config()
    model = GenixModel(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for loader_name, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(10)
            for k in range(10):
                X, Y = loader.get_batch()
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[loader_name] = losses.mean()
        model.train()
        return out

    # -----------------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------------
    t0 = time.time()
    for iter in range(max_iters):
        
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = train_loader.get_batch()

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds.")
    
    # Save the baseline model
    torch.save(model.state_dict(), 'genix_baseline.pt')
    print("Saved model checkpoint to genix_baseline.pt")

if __name__ == '__main__':
    train()
