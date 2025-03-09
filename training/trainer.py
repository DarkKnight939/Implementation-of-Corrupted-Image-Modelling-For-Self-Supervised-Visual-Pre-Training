import os
import torch
import tqdm
import numpy as np
import random
from torchvision.utils import save_image
from ..training.dataset import setup_data
from ..models.generator import Generator
from ..models.enhancer import CIMEnhancer
from ..training.CIM_trainer import CIMTrainer


def train_cim(
    num_epochs=15,
    batch_size=32,
    image_size=256,  # DALL-E expects 256x256
    d_model=384,     # Generator's BEIT dimension
    feature_dim=384, # Enhancer's feature dimension
    n_heads=6,
    n_layers=6,
    device='cuda',
    save_dir='./checkpoints',
    data_dir='./data',
    log_freq=10,
    save_freq=2,  # Save every 2 epochs
    task='respix',    # 'respix' or 'revdet'
    train_loader=None,  # Add optional data loader parameters
    test_loader=None
):
    """
    Main training function for Corrupted Image Modeling
    """
    print(f"\nInitializing CIM training with parameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"image_size: {image_size}")
    print(f"d_model: {d_model}")
    print(f"task: {task}")
    print(f"device: {device}\n")

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

    # Setup data if not provided
    if train_loader is None or test_loader is None:
        print("Setting up data loaders...")
        train_loader, test_loader = setup_data(
            batch_size=batch_size,
            image_size=image_size,
            data_dir=data_dir,
            device=device  # Pass device to setup_data
        )
        print(f"Created data loaders with {len(train_loader)} training batches")

    # Initialize models and ensure they're on the correct device
    print("\nInitializing models...")
    generator = Generator(
        device=device,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)  # Explicitly move to device

    enhancer = CIMEnhancer(
        feature_dim=feature_dim
    ).to(device)  # Explicitly move to device

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Enhancer parameters: {sum(p.numel() for p in enhancer.parameters())}")

    # Initialize trainer
    trainer = CIMTrainer(
        generator=generator,
        enhancer=enhancer,
        train_dataloader=train_loader,
        device=device
    )

    # Training loop
    print("\nStarting training loop...")
    best_gen_loss = float('inf')
    best_enh_loss = float('inf')

    epoch_progress = tqdm.tqdm(range(num_epochs), desc="Training Progress")
    for epoch in epoch_progress:
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training step
        avg_gen_loss, avg_enh_loss = trainer.train_epoch(epoch, task=task)

        # Update epoch progress bar
        epoch_progress.set_postfix({
            'gen_loss': f"{avg_gen_loss:.4f}",
            'enh_loss': f"{avg_enh_loss:.4f}"
        })

        print(f"Epoch {epoch+1} Summary:")
        print(f"Generator Loss: {avg_gen_loss:.4f}")
        print(f"Enhancer Loss: {avg_enh_loss:.4f}")

        # Save best models
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': trainer.gen_optim.state_dict(),
                'loss': avg_gen_loss,
            }, os.path.join(save_dir, 'best_generator.pt'))

        if avg_enh_loss < best_enh_loss:
            best_enh_loss = avg_enh_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': enhancer.state_dict(),
                'optimizer_state_dict': trainer.enh_optim.state_dict(),
                'loss': avg_enh_loss,
            }, os.path.join(save_dir, 'best_enhancer.pt'))

        # Regular checkpoints
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'enhancer_state_dict': enhancer.state_dict(),
                'gen_optimizer_state_dict': trainer.gen_optim.state_dict(),
                'enh_optimizer_state_dict': trainer.enh_optim.state_dict(),
                'gen_loss': avg_gen_loss,
                'enh_loss': avg_enh_loss,
            }, checkpoint_path)

        # Generate and save sample images
        if (epoch + 1) % log_freq == 0:
            generator.eval()
            enhancer.eval()
            with torch.no_grad():
                # Get a batch of test images
                test_batch = next(iter(test_loader))
                test_images = test_batch['image'].to(device)

                # Generate corrupted images
                corrupted_images, _, _ = generator(test_images)

                # Enhance corrupted images
                if task == 'respix':
                    enhanced_images, _ = enhancer(test_images, corrupted_images, task='respix')
                else:  # revdet
                    detection_map = enhancer(test_images, corrupted_images, task='revdet')
                    enhanced_images = corrupted_images  # Just for visualization

                # Save sample images
                for i in range(min(5, batch_size)):  # Save first 5 images
                    sample_path = os.path.join(save_dir, 'samples', f'epoch_{epoch+1}_sample_{i}.png')
                    # Create a grid of original, corrupted, and enhanced images
                    grid = torch.cat([
                        test_images[i].cpu(),
                        corrupted_images[i].cpu(),
                        enhanced_images[i].cpu()
                    ], dim=2)  # Concatenate horizontally
                    save_image(grid, sample_path, normalize=True)

    print("\nTraining completed!")
    return generator, enhancer

# Usage example:
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
        task='revdet'  # or 'revdet'
    )