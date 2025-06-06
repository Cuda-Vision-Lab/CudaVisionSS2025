import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # Encoder
        self.encoder = self.make_encoder()
        
        # Fully connected layers for mu and sigma
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_sigma = nn.Linear(2048, latent_dim)
        
        '''
        Also called projection layer, projects latent vector to  original dimensionality.
        maps the latent variable into a space compatible with the decoder input
        '''
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 2048),
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
        
    def forward(self, x):
        # Encode
        x_encoded = self.encoder(x)
        
        # Get mu and log_var
        mu = self.fc_mu(x_encoded)  # [batch_size, latent_dim]
        log_var = self.fc_sigma(x_encoded)  # [batch_size, latent_dim]
        
        # Reparameterization trick
        z = self.reparameterize(mu, log_var)  # [batch_size, latent_dim]
        
        # Project and reshape for decoder
        z = self.decoder_input(z)
        z = z.view(-1, 128, 4, 4)
        x_hat = self.decoder(z)
        
        return x_hat, (z, mu, log_var) 
    


# class CVAE(nn.Module):
#     def __init__(self, in_channels=3, latent_dim=256):  # increased latent dim
#         super(CVAE, self).__init__()
        
#         self.latent_dim = latent_dim
        
#         # Encoder with more channels and residual connections
#         self.encoder = nn.Sequential(
#             # Input: (3, 64, 64)
#             nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 4, 4)
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
            
#             nn.Flatten()  # 256 * 4 * 4 = 4096
#         )
        
#         # Fully connected layers for mu and sigma
#         self.fc_mu = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, latent_dim)
#         )
        
#         self.fc_sigma = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, latent_dim)
#         )
        
#         # Enhanced decoder input projection
#         self.decoder_input = nn.Sequential(
#             nn.Linear(latent_dim, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 4096),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Decoder with skip connections
#         self.decoder= nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
        
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),

#             nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # (3, 64, 64)
#             nn.Sigmoid()  # Changed to Tanh for better gradient flow
#         )
        
#     def reparameterize(self, mu, log_var):
#         """Reparameterization trick with temperature annealing"""
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
        
#     def forward(self, x):
#         # Encode
#         x_encoded = self.encoder(x)
        
#         # Get mu and log_var
#         mu = self.fc_mu(x_encoded)
#         log_var = self.fc_sigma(x_encoded)
        
#         # Reparameterization trick
#         z = self.reparameterize(mu, log_var)
        
#         # Project and reshape for decoder
#         z = self.decoder_input(z)
#         z = z.view(-1, 256, 4, 4)
#         x_hat = self.decoder(z)
         
#         return x_hat, (z, mu, log_var)