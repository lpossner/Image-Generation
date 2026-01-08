import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for Deep Convolutional GAN (DCGAN).

    Generates images from random noise vectors using transposed convolutional
    layers. Progressively upsamples from a latent vector to 64x64 images.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the DCGAN Generator.

        Args:
            num_colors (int): Number of color channels in output images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the input noise vector.
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Input: Z (latent vector) with size latent_dim
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=hidden_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 8),
            nn.ReLU(inplace=True),
            # State: (hidden_dim*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 4),
            nn.ReLU(inplace=True),
            # State: (hidden_dim*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim * 2),
            nn.ReLU(inplace=True),
            # State: (hidden_dim*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True),
            # State: (hidden_dim) x 32 x 32
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=num_colors, kernel_size=4, stride=2, padding=1, bias=False),
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


class Discriminator(nn.Module):
    """
    Discriminator network for Deep Convolutional GAN (DCGAN).

    Classifies images as real or fake using convolutional layers. Progressively
    downsamples 64x64 images to a single scalar prediction.
    """

    def __init__(self, num_colors, hidden_dim):
        """
        Initialize the DCGAN Discriminator.

        Args:
            num_colors (int): Number of color channels in input images (e.g., 3 for RGB).
            hidden_dim (int): Base number of channels in hidden layers (multiplied at each layer).
        """
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # Input: (num_colors) x 64 x 64
            nn.Conv2d(in_channels=num_colors, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            # No batch norm here according to DCGAN paper
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
            nn.Conv2d(in_channels=hidden_dim * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1
        )

    def forward(self, x):
        """
        Classify images as real or fake.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, num_colors, 64, 64).

        Returns:
            torch.Tensor: Predictions in range [0, 1], shape (batch_size,).
                         Values close to 1 indicate real images, close to 0 indicate fake.
        """
        return self.discriminator(x).view(-1).squeeze()


class DCGAN(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN).

    Combines a generator and discriminator for adversarial training. The generator
    learns to create realistic images while the discriminator learns to distinguish
    real from generated images.
    """

    def __init__(self, num_colors, latent_dim, hidden_dim):
        """
        Initialize the DCGAN model.

        Args:
            num_colors (int): Number of color channels in images (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the input noise vector.
            hidden_dim (int): Base number of channels in hidden layers.
        """
        super(DCGAN, self).__init__()
        self.generator = Generator(latent_dim=latent_dim, hidden_dim=hidden_dim, num_colors=num_colors)
        self.discriminator = Discriminator(hidden_dim=hidden_dim, num_colors=num_colors)
        self.apply(DCGAN.weights_init)

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
        Initialize network weights as suggested in the DCGAN paper.

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
