import torch
import numpy as np
import random
from training.trainer import train_cim

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the models
    generator, enhancer = train_cim(
        num_epochs=15,
        batch_size=32,
        device=device,
        task='revdet'  # or 'respix'
    )
