import torch
import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from ..utils.scheduler import ScheduledOptim


# Generator Trainer Implementation
class GeneratorTrainer:
    def __init__(
            self,
            generator,
            train_dataloader=None,
            test_dataloader=None,
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            warmup_steps=10000,
            log_freq=10,
            device='cuda'
    ):
        self.device = device
        self.generator = generator.to(device)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Optimizer for generator
        self.gen_optim = Adam(self.generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.gen_schedule = ScheduledOptim(self.gen_optim, self.generator.d_model, warmup_steps)

        self.criterion = MSELoss()  # For image reconstruction
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.generator.parameters()]))

    def train_step(self, batch):
        self.generator.train()

        # Move batch to device
        images = batch['image'].to(self.device)

        # Forward pass through generator
        corrupted_image, _, _ = self.generator(images)

        # Compute reconstruction loss
        loss = self.criterion(corrupted_image, images)

        # Update generator
        self.gen_schedule.zero_grad()
        loss.backward()
        self.gen_schedule.step_and_update_lr()

        return loss.item()

    def train_epoch(self, epoch):
        total_loss = 0

        # Main progress bar for the epoch
        epoch_iter = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"Epoch {epoch + 1}",
            total=len(self.train_data),
            leave=True,
            position=0
        )

        # Secondary progress bar for loss tracking
        loss_meter = tqdm.tqdm(
            total=0,
            desc="Loss",
            bar_format="{desc}: {postfix[0]:1.4f}",
            position=1,
            leave=True,
            postfix=[0]
        )

        data_iter = tqdm.tqdm(
            enumerate(self.train_data),
            desc=f"EP_train:{epoch}",
            total=len(self.train_data),
            bar_format="{l_bar}{r_bar}"
        )

        for i, batch in epoch_iter:
            loss = self.train_step(batch)
            total_loss += loss

            # Update the loss meter
            current_avg_loss = total_loss / (i + 1)
            loss_meter.postfix[0] = current_avg_loss
            loss_meter.update()

            if i % self.log_freq == 0:
                epoch_iter.set_postfix(loss=f"{current_avg_loss:.4f}")

        # Close progress bars
        epoch_iter.close()
        loss_meter.close()

        avg_loss = total_loss / len(self.train_data)
        return avg_loss
