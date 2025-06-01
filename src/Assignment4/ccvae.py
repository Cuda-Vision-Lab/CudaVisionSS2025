import torch
import torch.nn as nn
import torch.nn.functional as F

class CCVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, num_classes=3):
        super(CCVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = self.make_encoder()
        
        # Fully connected layers for mu and sigma
        # Added num_classes to account for class conditioning
        self.fc_mu = nn.Linear(2048 + num_classes, latent_dim)
        self.fc_sigma = nn.Linear(2048 + num_classes, latent_dim)
        
        # Decoder input layer (projection layer)
        # Takes latent vector and class embedding
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 2048),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = self.make_decoder()

    def make_encoder(self):
        return nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(self.in_channels, 16, kernel_size=4, stride=2, padding=1),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()  # 128 * 4 * 4 = 2048
        )
        
    def make_decoder(self):
        return nn.Sequential(
            # Input: (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(16, self.in_channels, kernel_size=4, stride=2, padding=1),  # (3, 64, 64)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
        """
        Encode the input image x conditioned on class label c
        Args:
            x: input image [batch_size, channels, height, width]
            c: class labels [batch_size]
        """
        # Convert class labels to one-hot encoding
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        
        # Encode image
        x_encoded = self.encoder(x)
        
        # Concatenate encoded image with class embedding
        x_c = torch.cat([x_encoded, c_onehot], dim=1)
        
        # Get mu and log_var
        mu = self.fc_mu(x_c)
        log_var = self.fc_sigma(x_c)
        
        return mu, log_var

    def decode(self, z, c):
        """
        Decode the latent vector z conditioned on class label c
        Args:
            z: latent vector [batch_size, latent_dim]
            c: class labels [batch_size]
        """
        # Convert class labels to one-hot encoding
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        
        # Concatenate latent vector with class embedding
        z_c = torch.cat([z, c_onehot], dim=1)
        
        # Project and reshape
        z_c = self.decoder_input(z_c)
        z_c = z_c.view(-1, 128, 4, 4)
        
        # Decode
        x_hat = self.decoder(z_c)
        return x_hat
        
    def forward(self, x, c):
        """
        Forward pass of the conditional VAE
        Args:
            x: input image [batch_size, channels, height, width]
            c: class labels [batch_size]
        """
        # Encode
        mu, log_var = self.encode(x, c)
        
        # Reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(z, c)
        
        return x_hat, (z, mu, log_var)

    def sample(self, num_samples, c):
        """
        Generate samples for given class labels
        Args:
            num_samples: number of samples to generate
            c: class labels [num_samples]
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        
        # Generate samples
        samples = self.decode(z, c)
        return samples
