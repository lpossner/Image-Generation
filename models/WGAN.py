
import torch.nn as nn

import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_colors, latent_dim, hidden_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Input: Z (latent vector) of shape (latent_dim, 1, 1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 16),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 16) x 4 x 4
            nn.ConvTranspose2d(in_channels=hidden_dim * 16, out_channels=hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 8) x 8 x 8
            nn.ConvTranspose2d(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 4) x 16 x 16
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 2) x 32 x 32
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=num_colors, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # Use sigmoid to output image pixel values between 0 and 1
            # Output: (num_colors) x 64 x 64
        )

    def forward(self, x):
        return self.generator(x)

class Critic(nn.Module):
    def __init__(self, num_colors, hidden_dim):
        super(Critic, self).__init__()
        self.discriminator = nn.Sequential(
            # No batch norm in critic for WGAN (violates Lipschitz constraint)
            # Input: (num_colors) x 64 x 64
            nn.Conv2d(in_channels=num_colors, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim) x 32 x 32
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*2) x 16 x 16
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*4) x 8 x 8
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*8) x 4 x 4
            nn.Conv2d(in_channels=hidden_dim * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            # Output: 1 (Wasserstein distance estimate)
        )

    def forward(self, x):
        return self.discriminator(x).view(-1)

    def clip_weights(self, clip_value=0.01):
        """Clip weights to enforce Lipschitz constraint as per WGAN paper"""
        for param in self.parameters():
            param.data.clamp_(-clip_value, clip_value)
    

class WGAN(nn.Module):

    def __init__(self, num_colors, latent_dim, hidden_dim):
        super(WGAN, self).__init__()
        self.generator = Generator(num_colors=num_colors, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.critic = Critic(num_colors=num_colors, hidden_dim=hidden_dim)
        self.apply(WGAN.weights_init)

    def forward(self, z):
        return self.generator(z)

    # Initialize weights as suggested in the DCGAN paper
    @staticmethod
    def weights_init(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
