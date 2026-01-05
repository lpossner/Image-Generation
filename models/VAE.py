import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_colors, latent_dim, hidden_dim):
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
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, num_colors, latent_dim, hidden_dim):
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
        z = self.fc(z)
        z = z.view(z.size(0), self.hidden_dim * 8, 4, 4)
        return self.decoder(z)


class VAE(nn.Module):

    def __init__(self, num_colors, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        ## Encoder
        self.encoder = Encoder(num_colors=num_colors, latent_dim=latent_dim, hidden_dim=hidden_dim)
        ## Decoder
        self.decoder = Decoder(latent_dim=latent_dim, num_colors=num_colors, hidden_dim=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
