import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CIMEnhancer(nn.Module):
    def __init__(self, num_classes: int = 2, feature_dim: int = 384):
        super().__init__()

        # Store feature_dim as class attribute
        self.feature_dim = feature_dim

        # Base ResNet-50 backbone
        resnet = models.resnet50(pretrained=False)
        resnet = resnet.float()

        # Feature Extraction Layers
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        # RESPIX Head for pixel-wise prediction
        self.respix_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        )

        # Enhanced REVDET head
        self.revdet_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )


    def compute_revdet_labels(self, original_tokens, generated_tokens):
        """
        Compute binary labels for token replacement detection
        Args:
            original_tokens: Original tokens from DALL-E encoder
            generated_tokens: Tokens after generator sampling
        Returns:
            Binary tensor indicating replaced tokens (1) vs original tokens (0)
        """
        return (original_tokens != generated_tokens).float()

    def sliding_window_normalization(self, x: torch.Tensor, window_size: int = 8) -> torch.Tensor:
        """
        Sliding window normalization with proper padding handling

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            window_size (int): Size of the normalization window

        Returns:
            torch.Tensor: Normalized image tensor of same size as input
        """
        b, c, h, w = x.shape

        # Ensure window size is odd for symmetric padding
        if window_size % 2 == 0:
            window_size += 1

        # Create averaging kernel for each channel
        kernel = torch.ones(1, 1, window_size, window_size).to(x.device) / (window_size * window_size)
        kernel = kernel.expand(c, 1, window_size, window_size)

        # Calculate padding
        pad = window_size // 2

        # Create a tensor to store local means and variances
        local_mean = torch.zeros_like(x)
        local_var = torch.zeros_like(x)

        for i in range(c):
            # Process each channel
            channel_data = x[:, i:i+1, :, :]

            # Compute mean using conv2d with reflection padding
            channel_mean = F.conv2d(
                F.pad(channel_data, (pad, pad, pad, pad), mode='reflect'),
                kernel[i:i+1],
                padding=0
            )
            local_mean[:, i:i+1, :, :] = channel_mean

            # Compute variance
            channel_sq = channel_data ** 2
            channel_sq_mean = F.conv2d(
                F.pad(channel_sq, (pad, pad, pad, pad), mode='reflect'),
                kernel[i:i+1],
                padding=0
            )
            local_var[:, i:i+1, :, :] = channel_sq_mean - channel_mean ** 2

        # Add epsilon for numerical stability and compute std
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))

        # Normalize
        normalized = (x - local_mean) / local_std

        # Verify output shape
        assert normalized.shape == x.shape, \
            f"Output shape {normalized.shape} doesn't match input shape {x.shape}"

        return normalized

    def forward(self, x, corrupted_x, task='respix'):
        """
        Args:
            x: Original image [B, 3, H, W]
            corrupted_x: Corrupted image [B, 3, H, W]
            task: 'respix' or 'revdet'
        """
        # Ensure inputs are float32
        x = x.float()
        corrupted_x = corrupted_x.float()

        # Print shapes for debugging
        print(f"Original image shape: {x.shape}")
        print(f"Corrupted image shape: {corrupted_x.shape}")

        # Verify input shapes
        assert x.size(1) == 3, f"Expected 3 channels for original image, got {x.size(1)}"
        assert corrupted_x.size(1) == 3, f"Expected 3 channels for corrupted image, got {corrupted_x.size(1)}"

        # Extract features from corrupted image only
        features = self.feature_extractor(corrupted_x)
        print(f"Features shape: {features.shape}")

        if task == 'respix':
            # Process original image for target
            normalized_target = self.sliding_window_normalization(x)
            print(f"Normalized target shape: {normalized_target.shape}")

            # Get predictions from corrupted image features
            predictions = self.respix_head(features)
            print(f"Predictions shape: {predictions.shape}")

            # Ensure predictions and target have same shape
            assert predictions.shape == normalized_target.shape, \
                f"Predictions shape {predictions.shape} doesn't match target shape {normalized_target.shape}"

            return predictions, normalized_target

        elif task == 'revdet':
            return self.revdet_head(features)

        raise ValueError(f"Unknown task: {task}")

    def compute_loss(self, predictions, target, task='respix', weights=None):
        """
        Enhanced loss computation with optional weighting
        """
        if task == 'respix':
            # For RESPIX, predictions and target should already be normalized
            l1_loss = nn.functional.l1_loss(predictions, target, reduction='none')
            l2_loss = nn.functional.mse_loss(predictions, target, reduction='none')

            if weights is not None:
                l1_loss = (l1_loss * weights).mean()
                l2_loss = (l2_loss * weights).mean()
            else:
                l1_loss = l1_loss.mean()
                l2_loss = l2_loss.mean()

            return 0.5 * l1_loss + 0.5 * l2_loss


        elif task == 'revdet':
            loss = nn.functional.binary_cross_entropy_with_logits(
                predictions,
                target,
                reduction='none'
            )

            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

            return loss