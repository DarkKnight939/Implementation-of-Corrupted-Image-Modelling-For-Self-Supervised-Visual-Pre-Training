# Set random seeds for reproducibility
import requests
import io
import os
import random
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from training.dataset import setup_data
from models.generator import Generator
from models.enhancer import CIMEnhancer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset (same as training)
batch_size = 32  # Adjust if needed
image_size = 256  # Ensure it matches training
data_dir = './data'

# Setup test data loader
_, test_loader = setup_data(
    batch_size=batch_size,
    image_size=image_size,
    data_dir=data_dir,
    device=device
)
print(f"Loaded test dataset with {len(test_loader)} batches")

def load_model(path, device):
    """Loads a PyTorch model from a local file or URL."""
    if path.startswith('http'):
        resp = requests.get(path, stream=True)
        resp.raw.decode_content = True  # Ensure raw data is decoded
        with io.BytesIO(resp.content) as buf:
            # Explicitly set weights_only=False to load the full model state
            return torch.load(buf, map_location=device, weights_only=False)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device, weights_only=False)  # Explicitly set weights_only=False

gen_checkpoint_path = '/kaggle/input/cim-model/pytorch/default/1/best_generator_respix.pt'
enh_checkpoint_path = '/kaggle/input/cim-model/pytorch/default/1/best_enhancer__respix.pt'

# Load Generator
generator = Generator(device=device, d_model=384, n_heads=6, n_layers=6).to(device)
gen_checkpoint = torch.load(gen_checkpoint_path, map_location=device)  # No weights_only
generator.load_state_dict(gen_checkpoint['model_state_dict'])  # Extract weights
generator.eval()
print("Loaded Generator model.")

# Load Enhancer
enhancer = CIMEnhancer(feature_dim=384).to(device)
enh_checkpoint = torch.load(enh_checkpoint_path, map_location=device)  # No weights_only
enhancer.load_state_dict(enh_checkpoint['model_state_dict'])  # Extract weights
enhancer.eval()
print("Loaded Enhancer model.")

output_dir = './test_results'
os.makedirs(output_dir, exist_ok=True)

# Run inference on test data
print("Running inference on test set...")
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):  # Use tqdm.tqdm
        test_images = batch['image'].to(device)
        # Generate corrupted images
        corrupted_images, _, _ = generator(test_images)

        # Enhance corrupted images
        enhanced_images = None
        task = 'revdet'
        if task == 'respix':
            enhanced_images, _ = enhancer(test_images, corrupted_images, task='respix')
        else:  # revdet
            detection_map = enhancer(test_images, corrupted_images, task='revdet')
            enhanced_images = corrupted_images  # Just for visualization

        # Save results (first 5 images per batch)
        for j in range(min(5, test_images.shape[0])):
            sample_path = os.path.join(output_dir, f"sample_{i}_{j}.png")
            grid = torch.cat([test_images[j].cpu(), corrupted_images[j].cpu(), enhanced_images[j].cpu()], dim=2)
            save_image(grid, sample_path, normalize=True)

print("Testing completed. Results saved in:", output_dir)

