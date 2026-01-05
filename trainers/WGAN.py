import math

from tqdm import tqdm

import torch

from torch.optim import AdamW


class WGANTrainer:

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
        self.model.train()
        epoch_loss_generator = 0.0
        epoch_loss_critic = 0.0
        total_samples = 0
        generator_updates = 0
        batch_count = 0
        with tqdm(self.data_loader) as progress:
            for images in progress:
                # Transfer images to device
                images_real = images.to(self.device)
                batch_size = images_real.size(0)

                ## Update critic
                self.optimizer.critic_optimizer.zero_grad()
                # Real and fake images
                predictions_real = self.model.critic(images_real)
                noise = torch.randn(size=(batch_size, self.latent_dim, 1, 1), device=self.device)
                images_fake = self.model.generator(noise)
                predictions_fake = self.model.critic(images_fake.detach()) # Detach to avoid updating the gradients of the generator
                # Compute loss
                wasserstein_distance = self.wasserstein_loss(predictions_real, predictions_fake)
                gradient_penalty = self.gradient_penalty(images_real, images_fake, batch_size)
                # Update critic
                loss_critic = wasserstein_distance + gradient_penalty
                loss_critic.backward()
                # Gradient clipping for critic
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=1.0)
                self.optimizer.critic_optimizer.step()

                ## Update generator (only every n_critic iterations)
                batch_count += 1
                if batch_count % self.n_critic == 0:
                    self.optimizer.generator_optimizer.zero_grad()
                    # Generate fresh fake images for generator training
                    noise = torch.randn(size=(batch_size, self.latent_dim, 1, 1), device=self.device)
                    fresh_fake_images = self.model.generator(noise)
                    predictions_generator = self.model.critic(fresh_fake_images)
                    loss_generator = -torch.mean(predictions_generator)
                    loss_generator.backward()
                    # Gradient clipping for generator
                    torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
                    self.optimizer.generator_optimizer.step()
                    # Update generator metrics
                    generator_updates += 1
                    epoch_loss_generator += (
                        (loss_generator.item() - epoch_loss_generator) / generator_updates
                    )
                    self.metrics["batch_loss/generator"] = loss_generator.item()
                    self.metrics["epoch_loss/generator"] = epoch_loss_generator

                ## Log metrics
                total_samples += batch_size
                epoch_loss_critic += (
                    (loss_critic.item() - epoch_loss_critic)
                    * batch_size
                    / total_samples
                )
                self.metrics["batch_loss/critic"] = loss_critic.item()
                self.metrics["epoch_loss/critic"] = epoch_loss_critic
                self.metrics["wasserstein_distance"] = wasserstein_distance.item()
                self.logger.log_metrics(self.metrics, epoch, epochs)
                self.logger.update_progress(progress, self.metrics, epoch, epochs)

        return self.metrics
    
    def gradient_penalty(self, images_real, images_fake, batch_size):
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device).expand_as(images_real)
        images_interpolated = epsilon * images_real + (1 - epsilon) * images_fake.detach()
        images_interpolated.requires_grad_(True)
        
        predictions_interpolated = self.model.critic(images_interpolated)
        gradients = torch.autograd.grad(
            outputs=predictions_interpolated,
            inputs=images_interpolated,
            grad_outputs=torch.ones(predictions_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        return self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

    def wasserstein_loss(self, predictions_real, predictions_fake):
        return torch.mean(predictions_fake) - torch.mean(predictions_real)


class Optimizer:

    def __init__(self, generator_optimizer, critic_optimizer):
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
    
    def state_dict(self):
        return {
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
            self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
