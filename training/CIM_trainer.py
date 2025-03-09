import torch
import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from ..utils.scheduler import ScheduledOptim

class CIMTrainer:
    def __init__(
        self,
        generator,
        enhancer,
        train_dataloader,
        device='cuda',
        lr=1.5e-3,
        weight_decay=0.05,
        warmup_steps=10,
        mask_ratio=0.55  # ~110 tokens out of 196 as per paper
    ):
        self.device = device
        self.generator = generator.to(device)
        self.enhancer = enhancer.to(device)
        self.train_data = train_dataloader
        self.mask_ratio = mask_ratio

        # Separate optimizers for generator (BEiT) and enhancer
        self.gen_optim = Adam(
            [p for p in self.generator.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )
        self.enh_optim = Adam(self.enhancer.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate schedulers
        self.gen_schedule = ScheduledOptim(self.gen_optim, generator.d_model, warmup_steps)
        self.enh_schedule = ScheduledOptim(self.enh_optim, enhancer.feature_dim, warmup_steps)

    def compute_beit_loss(self, predicted_tokens, golden_tokens, mask_indices):
        """
        Compute MIM loss for generator (BEiT) as per paper
        """
        # Only compute loss for masked positions
        pred_masked = predicted_tokens[:, mask_indices]
        gold_masked = golden_tokens[:, mask_indices]

        # Cross entropy loss for token prediction
        return F.cross_entropy(pred_masked, gold_masked)

    def train_step(self, batch, task='respix'):
        self.generator.train()
        self.enhancer.train()

        # Move batch to device here
        images = batch['image'].to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Get batch size from images tensor
        B = images.size(0)

        # 1. Generator (BEiT) Forward Pass
        with torch.no_grad():
            golden_tokens = self.generator.tokenizer.encode(images)  # [B, 8192, H/8, W/8]

        # Properly reshape tokens
        B, C, H, W = golden_tokens.shape
        golden_tokens = golden_tokens.permute(0, 2, 3, 1)  # [B, H/8, W/8, 8192]
        golden_tokens = golden_tokens.reshape(B, H*W, C)   # [B, (H/8)*(W/8), 8192]

        # Create patch embeddings
        patch_embeddings = self.generator.patch_embedding(golden_tokens)  # [B, (H/8)*(W/8), d_model]
        patch_embeddings = patch_embeddings + self.generator.pos_embedding(patch_embeddings)

        # Random masking (50-60% as per paper)
        num_patches = patch_embeddings.size(1)
        num_mask = int(0.55 * num_patches)
        mask_indices = torch.randperm(num_patches, device=self.device)[:num_mask]
        bool_masked_pos = torch.zeros(num_patches, dtype=torch.bool, device=self.device)
        bool_masked_pos[mask_indices] = True

        # Forward through transformer layers
        x = patch_embeddings.clone()
        for layer in self.generator.transformer_layers:
            x = layer(x)

        # Token prediction
        token_logits = self.generator.token_predictor(x)  # [B, num_patches, 8192]

        # Get the target tokens for masked positions
        target_tokens = golden_tokens.argmax(dim=-1)  # Convert one-hot to indices [B, num_patches]
        masked_logits = token_logits[:, mask_indices, :]  # [B, num_mask, 8192]
        masked_targets = target_tokens[:, mask_indices]   # [B, num_mask]

        # Compute generator loss
        generator_loss = F.cross_entropy(
            masked_logits.reshape(-1, 8192),  # [B*num_mask, 8192]
            masked_targets.reshape(-1)         # [B*num_mask]
        )

        # Sample new tokens for masked positions
        with torch.no_grad():
            sampled_probs = F.softmax(masked_logits / self.generator.temperature, dim=-1)
            sampled_tokens = torch.multinomial(
                sampled_probs.view(-1, 8192),
                1
            ).view(B, -1)  # [B, num_mask]

        # Create corrupted tokens
        corrupted_tokens = golden_tokens.clone()
        corrupted_tokens[:, mask_indices] = F.one_hot(sampled_tokens, num_classes=8192).float()

        # Reshape back to image format
        corrupted_tokens = corrupted_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Generate corrupted image
        with torch.no_grad():
            corrupted_image = self.generator.tokenizer.decode(corrupted_tokens)
                # 2. Enhancer Forward Pass
        if task == 'respix':
            predictions, normalized_target = self.enhancer(
                images,  # original images
                corrupted_image,  # corrupted images
                task='respix'
            )
            enhancer_loss = self.enhancer.compute_loss(
                predictions,
                normalized_target,
                task='respix'
            )
        else:  # revdet
            predictions = self.enhancer(
                images,
                corrupted_image,
                task='revdet'
            )
            replacement_labels = bool_masked_pos.unsqueeze(0).expand(B, -1).float()
            enhancer_loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), replacement_labels.mean(dim=1) )
        # Update Generator
        self.gen_optim.zero_grad()
        generator_loss.backward()
        self.gen_optim.step()

        # Update Enhancer
        self.enh_optim.zero_grad()
        enhancer_loss.backward()
        self.enh_optim.step()

        return {
            'generator_loss': generator_loss.item(),
            'enhancer_loss': enhancer_loss.item()
        }

    def train_epoch(self, epoch, task='respix'):
        total_gen_loss = 0
        total_enh_loss = 0

        progress_bar = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"Epoch {epoch+1}",
            total=len(self.train_data)
        )

        for i, batch in progress_bar:
            losses = self.train_step(batch, task)
            total_gen_loss += losses['generator_loss']
            total_enh_loss += losses['enhancer_loss']

            # Update progress bar
            avg_gen_loss = total_gen_loss / (i + 1)
            avg_enh_loss = total_enh_loss / (i + 1)
            progress_bar.set_postfix({
                'gen_loss': f"{avg_gen_loss:.4f}",
                'enh_loss': f"{avg_enh_loss:.4f}"
            })

        return avg_gen_loss, avg_enh_loss