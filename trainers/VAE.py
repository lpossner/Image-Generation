import math

from tqdm import tqdm

import torch

import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class VAETrainer:
    """
    Trainer for Variational Autoencoder (VAE) model.

    Handles training loop, loss computation, and metric logging for VAE training.
    Uses AdamW optimizer and combines reconstruction loss with KL divergence.
    """

    def __init__(self, model, device, data_loader, checkpoint_name, logger, lr):
        """
        Initialize the VAE trainer.

        Args:
            model (VAE): VAE model to train.
            device (torch.device): Device to train on (CPU/CUDA/MPS).
            data_loader (DataLoader): DataLoader providing training batches.
            checkpoint_name (str): Name for saving checkpoints.
            logger (Logger): Logger for metrics and progress tracking.
            lr (float): Learning rate for optimizer.
        """
        self.model = model.to(device)
        self.optimizer = AdamW(params=model.parameters(), lr=lr)
        self.device = device
        self.data_loader = data_loader
        self.checkpoint_name = checkpoint_name
        self.logger = logger
        self.metrics = {"epoch_loss": math.inf,
                        "batch_loss": math.inf
                        }

    def train_epoch(self, epoch, epochs):
        """
        Train the VAE for one epoch.

        Args:
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.

        Returns:
            dict: Dictionary containing training metrics including epoch_loss, batch_loss,
                  reconstruction_loss, and kl_loss.
        """
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        with tqdm(self.data_loader) as progress:
            for images in progress:
                ## Update VAE
                images = images.to(self.device)
                batch_size = images.size(0)
                self.optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(images)
                
                # DEBUG
                reconstruction_loss = self.reconstruction_loss(reconstructed, images, mu, logvar)
                kl_loss = self.kl_loss(reconstructed, images, mu, logvar)
                loss = reconstruction_loss + kl_loss
                # DEBUG
                
                # loss = self.loss(reconstructed, images, mu, logvar)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Step optimizer
                self.optimizer.step()
                ## Log metrics
                total_samples += batch_size
                epoch_loss += (
                    (loss.item() - epoch_loss) * batch_size / total_samples
                )
                self.metrics["epoch_loss"] = epoch_loss
                self.metrics["batch_loss"] = loss.item()

                # DEBUG
                self.metrics["reconstruction_loss"] = reconstruction_loss.item()
                self.metrics["kl_loss"] = kl_loss.item()
                # DEBUG

                self.logger.log_metrics(self.metrics, epoch, epochs)
                self.logger.update_progress(progress, self.metrics, epoch, epochs)

        return self.metrics

    def loss(self, reconstructed_x, x, mu, logvar):
        """
        Compute total VAE loss (reconstruction + KL divergence).

        Args:
            reconstructed_x (torch.Tensor): Reconstructed images from decoder.
            x (torch.Tensor): Original input images.
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log-variance of latent distribution.

        Returns:
            torch.Tensor: Total VAE loss (scalar).
        """
        # Reconstruction loss
        reconstructed_loss = (
            F.binary_cross_entropy(reconstructed_x, x, reduction="none")
            .sum(dim=[1, 2, 3])
            .mean()
        )
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return reconstructed_loss + kl_loss

    def reconstruction_loss(self, reconstructed_x, x, mu, logvar):
        """
        Compute reconstruction loss using binary cross-entropy.

        Args:
            reconstructed_x (torch.Tensor): Reconstructed images from decoder.
            x (torch.Tensor): Original input images.
            mu (torch.Tensor): Mean of latent distribution (unused, for signature compatibility).
            logvar (torch.Tensor): Log-variance of latent distribution (unused, for signature compatibility).

        Returns:
            torch.Tensor: Reconstruction loss (scalar).
        """
        return (
            F.binary_cross_entropy(reconstructed_x, x, reduction="none")
            .sum(dim=[1, 2, 3])
            .mean()
        )

    def kl_loss(self, reconstructed_x, x, mu, logvar):
        """
        Compute KL divergence loss between latent distribution and standard normal.

        Args:
            reconstructed_x (torch.Tensor): Reconstructed images (unused, for signature compatibility).
            x (torch.Tensor): Original input images (unused, for signature compatibility).
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log-variance of latent distribution.

        Returns:
            torch.Tensor: KL divergence loss (scalar).
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
