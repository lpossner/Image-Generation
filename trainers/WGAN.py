import math

from tqdm import tqdm

import torch

from torch.optim import AdamW


class WGANTrainer:
    """
    Trainer for Wasserstein GAN (WGAN) with Gradient Penalty.

    Implements WGAN-GP training with critic updated multiple times per generator
    update. Uses gradient penalty to enforce Lipschitz constraint instead of
    weight clipping.
    """

    def __init__(
        self,
        model,
        device,
        data_loader,
        checkpoint_name,
        logger,
        lr_generator,
        lr_critic,
        beta1,
        beta2,
        latent_dim,
        lambda_gp,
        n_critic,
    ):
        """
        Initialize the WGAN trainer.

        Args:
            model (WGAN): WGAN model to train.
            device (torch.device): Device to train on (CPU/CUDA/MPS).
            data_loader (DataLoader): DataLoader providing training batches.
            checkpoint_name (str): Name for saving checkpoints.
            logger (Logger): Logger for metrics and progress tracking.
            lr_generator (float): Learning rate for generator optimizer.
            lr_critic (float): Learning rate for critic optimizer.
            beta1 (float): Beta1 parameter for Adam optimizer.
            beta2 (float): Beta2 parameter for Adam optimizer.
            latent_dim (int): Dimensionality of noise vector for generator.
            lambda_gp (float): Weight for gradient penalty term.
            n_critic (int): Number of critic updates per generator update.
        """
        self.model = model.to(device)
        generator_optimizer = AdamW(
            model.generator.parameters(), lr=lr_generator, betas=(beta1, beta2)
        )
        critic_optimizer = AdamW(
            model.critic.parameters(), lr=lr_critic, betas=(beta1, beta2)
        )
        self.optimizer = Optimizer(generator_optimizer, critic_optimizer)
        self.device = device
        self.data_loader = data_loader
        self.checkpoint_name = checkpoint_name
        self.logger = logger
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.metrics = {
            "batch_loss/critic": math.inf,
            "batch_loss/generator": math.inf,
            "epoch_loss/critic": math.inf,
            "epoch_loss/generator": math.inf,
        }

    def train_epoch(self, epoch, epochs):
        """
        Train the WGAN for one epoch.

        Updates critic multiple times (n_critic) per generator update. Critic
        is trained to maximize Wasserstein distance with gradient penalty.
        Generator is trained to minimize Wasserstein distance.

        Args:
            epoch (int): Current epoch number (0-indexed).
            epochs (int): Total number of epochs.

        Returns:
            dict: Dictionary containing training metrics including batch_loss and
                  epoch_loss for generator and critic, plus wasserstein_distance.
        """
        self.model.train()
        epoch_loss_generator = 0.0
        epoch_loss_critic = 0.0
        total_samples_critic = 0
        total_samples_generator = 0
        batch_count = 0
        with tqdm(self.data_loader) as progress:
            for images in progress:
                # Transfer images to device
                images_real = images.to(self.device)
                batch_size = images_real.size(0)

                ## Update critic
                self.optimizer.critic_optimizer.zero_grad()
                # Real and fake images
                predictions_critic_real = self.model.critic(images_real)
                noise = torch.randn(
                    size=(batch_size, self.latent_dim, 1, 1), device=self.device
                )
                images_fake = self.model.generator(noise)
                predictions_critic_fake = self.model.critic(
                    images_fake.detach()
                )  # Detach to avoid updating the gradients of the generator
                # Compute loss
                wasserstein_distance = self.wasserstein_loss(
                    predictions_critic_real, predictions_critic_fake
                )
                gradient_penalty = self.gradient_penalty(
                    images_real, images_fake, batch_size
                )
                # Update critic
                loss_critic = wasserstein_distance + gradient_penalty
                loss_critic.backward()
                # Gradient clipping for critic
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), max_norm=1.0
                )
                self.optimizer.critic_optimizer.step()
                # Update critic metrics
                total_samples_critic += batch_size
                epoch_loss_critic += (
                    (loss_critic.item() - epoch_loss_critic)
                    * batch_size
                    / total_samples_critic
                )

                ## Update generator (only every n_critic iterations)
                batch_count += 1
                if batch_count % self.n_critic == 0:
                    self.optimizer.generator_optimizer.zero_grad()
                    # Generate fresh fake images for generator training
                    noise = torch.randn(
                        size=(batch_size, self.latent_dim, 1, 1), device=self.device
                    )
                    images_fake = self.model.generator(noise)
                    predictions_generator_fake = self.model.critic(images_fake)
                    loss_generator = -torch.mean(predictions_generator_fake)
                    loss_generator.backward()
                    # Gradient clipping for generator
                    torch.nn.utils.clip_grad_norm_(
                        self.model.generator.parameters(), max_norm=1.0
                    )
                    self.optimizer.generator_optimizer.step()
                    # Update generator metrics
                    total_samples_generator += batch_size
                    epoch_loss_generator += (
                        (loss_generator.item() - epoch_loss_generator)
                        * batch_size
                        / total_samples_generator
                    )
                    self.metrics["batch_loss/generator"] = loss_generator.item()
                    self.metrics["epoch_loss/generator"] = epoch_loss_generator

                ## Log metrics
                self.metrics["batch_loss/critic"] = loss_critic.item()
                self.metrics["epoch_loss/critic"] = epoch_loss_critic
                self.metrics["wasserstein_distance"] = wasserstein_distance.item()
                self.logger.log_metrics(self.metrics, epoch, epochs)
                self.logger.update_progress(progress, self.metrics, epoch, epochs)

        return self.metrics

    def gradient_penalty(self, images_real, images_fake, batch_size):
        """
        Compute gradient penalty for WGAN-GP.

        Enforces 1-Lipschitz constraint by penalizing gradients with norm != 1
        on interpolated images between real and fake samples.

        Args:
            images_real (torch.Tensor): Real images from dataset.
            images_fake (torch.Tensor): Fake images from generator.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Gradient penalty loss term (scalar).
        """
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device).expand_as(
            images_real
        )
        images_interpolated = (
            epsilon * images_real + (1 - epsilon) * images_fake.detach()
        )
        images_interpolated.requires_grad_(True)

        predictions_interpolated = self.model.critic(images_interpolated)
        gradients = torch.autograd.grad(
            outputs=predictions_interpolated,
            inputs=images_interpolated,
            grad_outputs=torch.ones(
                predictions_interpolated.size(), device=self.device
            ),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        return self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

    def wasserstein_loss(self, predictions_real, predictions_fake):
        """
        Compute Wasserstein distance estimate (critic loss).

        Args:
            predictions_real (torch.Tensor): Critic scores for real images.
            predictions_fake (torch.Tensor): Critic scores for fake images.

        Returns:
            torch.Tensor: Wasserstein distance estimate (scalar). Negative values
                         indicate critic loss.
        """
        return torch.mean(predictions_fake) - torch.mean(predictions_real)


class Optimizer:
    """
    Wrapper class for dual optimizers (generator and critic).

    Provides a unified interface for state management of both optimizers.
    """

    def __init__(self, generator_optimizer, critic_optimizer):
        """
        Initialize the optimizer wrapper.

        Args:
            generator_optimizer (torch.optim.Optimizer): Optimizer for generator.
            critic_optimizer (torch.optim.Optimizer): Optimizer for critic.
        """
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

    def state_dict(self):
        """
        Get state dictionary for both optimizers.

        Returns:
            dict: Dictionary containing state_dict for generator and critic optimizers.
        """
        return {
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Load state dictionary for both optimizers.

        Args:
            state_dict (dict): Dictionary containing state_dict for both optimizers.
        """
        self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
