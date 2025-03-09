from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch

class BEITDatasetCifar(Dataset):
    def __init__(self, cifar_data, device):
        self.cifar_data = cifar_data
        self.device = device
        self.transform = T.Compose([
            T.RandomResizedCrop((256, 256)),  # Add random resized crop
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.cifar_data)

    def __getitem__(self, idx):
        image, _ = self.cifar_data[idx]
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        # Don't move to device here
        return {'image': image}

def setup_data(
    batch_size=32,
    image_size=256,
    num_workers=4,
    data_dir="./data",
    device='cuda'
):
    # Load CIFAR-10 dataset with minimal transforms
    cifar_train = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )

    cifar_test = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )

    # Wrap with our custom dataset
    train_dataset = BEITDatasetCifar(cifar_train, device)
    test_dataset = BEITDatasetCifar(cifar_test, device)

    # Create data loaders with num_workers=0 if using CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if device == 'cuda' else num_workers,  # Set to 0 for CUDA
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device == 'cuda' else num_workers,  # Set to 0 for CUDA
        pin_memory=True
    )

    return train_loader, test_loader
