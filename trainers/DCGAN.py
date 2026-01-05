import math

from tqdm import tqdm

import torch

from torch.optim import AdamW

import torch.nn.functional as F


class DCGANTrainer:

    def __init__(
        self,
        model,
        device,
        data_loader,
        checkpoint_name,
        logger,
        lr_generator,
        lr_discriminator,
        beta1,
        beta2,
        latent_dim,
    ):
        self.model = model.to(device)
        generator_optimizer = AdamW(
            model.generator.parameters(), lr=lr_generator, betas=(beta1, beta2)
        )
        discriminator_optimizer = AdamW(
            model.discriminator.parameters(), lr=lr_discriminator, betas=(beta1, beta2)
        )
        self.optimizer = Optimizer(generator_optimizer, discriminator_optimizer)
        self.device = device
        self.data_loader = data_loader
        self.checkpoint_name = checkpoint_name
        self.logger = logger
        self.latent_dim = latent_dim
        self.metrics = {
            "batch_loss/discriminator": math.inf,
            "batch_loss/generator": math.inf,
            "epoch_loss/discriminator": math.inf,
            "epoch_loss/generator": math.inf,
        }

    def train_epoch(self, epoch, epochs):
        self.model.train()
        epoch_loss_generator = 0.0
        epoch_loss_discriminator = 0.0
        total_samples = 0
        with tqdm(self.data_loader) as progress:
            for images in progress:
                ## Update Discriminator
                # Prepare
                self.optimizer.discriminator_optimizer.zero_grad()
                images_real = images.to(self.device)
                batch_size = images_real.size(0)
                # Show the discriminator real images
                labels_real = torch.full(
                    size=(batch_size,),
                    fill_value=0.9,
                    dtype=torch.float,
                    device=self.device,
                )
                predictions_real = self.model.discriminator(images_real)
                loss_real = self.loss(predictions_real, labels_real)
                loss_real.backward()

                # Generate fake images
                noise = torch.randn(
                    size=(batch_size, self.latent_dim, 1, 1), device=self.device
                )
                fake_images = self.model.generator(noise)

                # Show the discriminator fake images
                labels_fake = torch.full(
                    size=(batch_size,),
                    fill_value=0.0,
                    dtype=torch.float,
                    device=self.device,
                )
                predictions_fake = self.model.discriminator(
                    fake_images.detach()
                )  # Detach to avoid updating the generator gradients here
                loss_fake = self.loss(predictions_fake, labels_fake)
                loss_fake.backward()

                # Combine loss and step optimizer
                loss_discriminator = loss_real + loss_fake
                # Gradient clipping for discriminator
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                self.optimizer.discriminator_optimizer.step()

                ## Update generator
                self.optimizer.generator_optimizer.zero_grad()
                # Generate new fake images for generator training
                noise = torch.randn(
                    size=(batch_size, self.latent_dim, 1, 1), device=self.device
                )
                fake_images = self.model.generator(noise)

                # Train the generator with predictions of the discriminator
                predictions_generator = self.model.discriminator(fake_images)

                # Create labels for generator (wants discriminator to predict 1.0)
                labels_generator = torch.full(
                    size=(batch_size,),
                    fill_value=1.0,
                    dtype=torch.float,
                    device=self.device,
                )

                # Compute loss and step optimizer
                loss_generator = self.loss(predictions_generator, labels_generator)
                loss_generator.backward()
                # Gradient clipping for generator
                torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
                self.optimizer.generator_optimizer.step()

                ## Log metrics
                total_samples += batch_size
                epoch_loss_generator += (
                    (loss_generator.item() - epoch_loss_generator)
                    * batch_size
                    / total_samples
                )
                epoch_loss_discriminator += (
                    (loss_discriminator.item() - epoch_loss_discriminator)
                    * batch_size
                    / total_samples
                )
                self.metrics["batch_loss/generator"] = loss_generator.item()
                self.metrics["batch_loss/discriminator"] = loss_discriminator.item()
                self.metrics["epoch_loss/generator"] = epoch_loss_generator
                self.metrics["epoch_loss/discriminator"] = epoch_loss_discriminator
                self.logger.log_metrics(self.metrics, epoch, epochs)
                self.logger.update_progress(progress, self.metrics, epoch, epochs)

        return self.metrics
    
    def loss(self, predictions, labels):
        return F.binary_cross_entropy(predictions, labels)
                

class Optimizer:

    def __init__(self, generator_optimizer, discriminator_optimizer):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
    
    def state_dict(self):
        return {
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
            self.generator_optimizer.load_state_dict(state_dict["generator_optimizer"])
            self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
