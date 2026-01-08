import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network for Variational Autoencoder (VAE).

    Maps input images to latent space distributions (mean and log-variance)
    using convolutional layers. The encoder progressively downsamples 64x64
    images to a latent representation.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the VAE Encoder.

        Args:
            num_colors (int): Number of color channels in input images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input: (num_colors) x 64 x 64
            nn.Conv2d(in_channels=num_colors, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=True),
            # No batch norm on first layer
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim) x 32 x 32
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*2) x 16 x 16
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*4) x 8 x 8
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State: (hidden_dim*8) x 4 x 4
        )
        # State: (hidden_dim*8) x 4 x 4
        self.flatten = nn.Flatten()
        # State: hidden_dim*8*4*4
        self.fc_mu = nn.Linear(in_features=hidden_dim * 8 * 4 * 4, out_features=latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(in_features=hidden_dim * 8 * 4 * 4, out_features=latent_dim)  # Log variance of latent space
        # Output: latent_dim

    def forward(self, x):
        """
        Encode input images to latent space distributions.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, num_colors, 64, 64).

        Returns:
            tuple: (mu, logvar) where:
                - mu (torch.Tensor): Mean of latent distribution, shape (batch_size, latent_dim).
                - logvar (torch.Tensor): Log-variance of latent distribution, shape (batch_size, latent_dim).
        """
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for Variational Autoencoder (VAE).

    Maps latent space vectors back to images using transposed convolutional layers.
    The decoder progressively upsamples from latent representation to 64x64 images.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the VAE Decoder.

        Args:
            num_colors (int): Number of color channels in output images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Linear layer to project latent vector to feature map
        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_dim * 8 * 4 * 4)
        # State after reshape: (hidden_dim * 8) x 4 x 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 4) x 8 x 8
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # State: (hidden_dim * 2) x 16 x 16
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # State: (hidden_dim) x 32 x 32
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=num_colors, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # Use sigmoid to output image pixel values between 0 and 1
            # Output: (num_colors) x 64 x 64
        )

    def forward(self, z):
        """
        Decode latent vectors to images.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, num_colors, 64, 64).
        """
        z = self.fc(z)
        z = z.view(z.size(0), self.hidden_dim * 8, 4, 4)
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for image generation.

    Combines an encoder and decoder to learn a probabilistic latent representation
    of images. Uses the reparameterization trick to enable backpropagation through
    stochastic sampling.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the VAE model.

        Args:
            num_colors (int): Number of color channels in images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Base number of channels in hidden layers.
        """
        super(VAE, self).__init__()
        ## Encoder
        self.encoder = Encoder(num_colors=num_colors, latent_dim=latent_dim, hidden_dim=hidden_dim)
        ## Decoder
        self.decoder = Decoder(latent_dim=latent_dim, num_colors=num_colors, hidden_dim=hidden_dim)

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample from N(mu, var).

        Enables backpropagation through stochastic sampling by expressing
        the random variable as a deterministic function of mu, logvar, and
        independent noise epsilon.

        Args:
            mu (torch.Tensor): Mean of the latent distribution, shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance of the latent distribution, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Sampled latent vectors, shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encode images to latent distributions.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, num_colors, 64, 64).

        Returns:
            tuple: (mu, logvar) representing the latent distribution parameters.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent vectors to images.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, num_colors, 64, 64).
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Encodes input images, samples from the latent distribution using the
        reparameterization trick, and decodes to reconstruct the images.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, num_colors, 64, 64).

        Returns:
            tuple: (reconstructed, mu, logvar) where:
                - reconstructed (torch.Tensor): Reconstructed images, shape (batch_size, num_colors, 64, 64).
                - mu (torch.Tensor): Mean of latent distribution, shape (batch_size, latent_dim).
                - logvar (torch.Tensor): Log-variance of latent distribution, shape (batch_size, latent_dim).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def generate(self, z):
        """
        Generate images from latent vectors.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, num_colors, 64, 64).
        """
        return self.decoder(z)
