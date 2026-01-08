
import torch.nn as nn

import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for Wasserstein GAN (WGAN).

    Generates images from random noise vectors using transposed convolutional
    layers. Uses a deeper architecture than DCGAN to improve image quality.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the WGAN Generator.

        Args:
            num_colors (int): Number of color channels in output images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the input noise vector.
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
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
        """
        Generate images from noise vectors.

        Args:
            x (torch.Tensor): Noise vectors of shape (batch_size, latent_dim, 1, 1).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, num_colors, 64, 64).
        """
        return self.generator(x)

class Critic(nn.Module):
    """
    Critic network for Wasserstein GAN (WGAN).

    Estimates the Wasserstein distance between real and generated image distributions.
    Unlike a discriminator, outputs unbounded real values rather than probabilities.
    Does not use batch normalization to satisfy the Lipschitz constraint.
    """

    def __init__(self, num_colors, hidden_dim):
        """
        Initialize the WGAN Critic.

        Args:
            num_colors (int): Number of color channels in input images (e.g., 3 for RGB).
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
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
        """
        Compute critic score for images.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, num_colors, 64, 64).

        Returns:
            torch.Tensor: Critic scores (unbounded real values), shape (batch_size,).
                         Higher values indicate more realistic images.
        """
        return self.critic(x).view(-1)

    def clip_weights(self, clip_value=0.01):
        """
        Clip critic weights to enforce Lipschitz constraint.

        Weight clipping is used in the original WGAN paper to satisfy the
        Lipschitz continuity requirement. Modern implementations typically
        use gradient penalty instead.

        Args:
            clip_value (float): Maximum absolute value for weights. Default: 0.01.
        """
        for param in self.parameters():
            param.data.clamp_(-clip_value, clip_value)
    

class WGAN(nn.Module):
    """
    Wasserstein Generative Adversarial Network (WGAN).

    Combines a generator and critic for adversarial training. Uses Wasserstein
    distance instead of Jensen-Shannon divergence for more stable training.
    The critic estimates the Wasserstein distance between real and generated
    distributions.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the WGAN model.

        Args:
            num_colors (int): Number of color channels in images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the input noise vector.
            hidden_dim (int): Base number of channels in hidden layers.
        """
        super(WGAN, self).__init__()
        self.generator = Generator(num_colors=num_colors, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.critic = Critic(num_colors=num_colors, hidden_dim=hidden_dim)
        self.apply(WGAN.weights_init)

    def forward(self, z):
        """
        Generate images from noise vectors.

        Args:
            z (torch.Tensor): Noise vectors of shape (batch_size, latent_dim, 1, 1).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, num_colors, 64, 64).
        """
        return self.generator(z)

    def generate(self, z):
        """
        Generate images from noise vectors (alias for forward).

        Args:
            z (torch.Tensor): Noise vectors of shape (batch_size, latent_dim, 1, 1).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, num_colors, 64, 64).
        """
        return self.generator(z)

    @staticmethod
    def weights_init(model):
        """
        Initialize network weights following DCGAN recommendations.

        Convolutional layers are initialized from N(0, 0.02) and batch
        normalization weights from N(1.0, 0.02) with biases set to 0.

        Args:
            model (nn.Module): Model or layer to initialize.
        """
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
