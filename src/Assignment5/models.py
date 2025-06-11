import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import os

class ConvBlock(nn.Module):
    """
    Simple convolutional block: Conv + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
            
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """
    Simple convolutional block: ConvTranspose + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y
    

class Generator(nn.Module):
    """
    Fully convolutional generator for both conditional and unconditional modes.
    Unconditional mode: (B, latent_dim, 1, 1) --> (B, num_channels, 64, 64)
    Conditional mode: (B, latent_dim + num_classes, 1, 1) --> (B, num_channels, 64, 64)
    """
    def __init__(self, latent_dim=128, num_channels=3, base_channels=64, num_classes=3, conditioned=False):
        """ Model initializer """
        super().__init__()

        self.latent_dim = latent_dim
        self.conditioned = conditioned  

        if conditioned:
            # Embedding layer for class labels (used in conditional mode)
            self.label_embedding = nn.Embedding(num_classes, num_classes)
            self.num_classes = num_classes
        
        layers = []
        in_channels= latent_dim if not conditioned else latent_dim + num_classes

        for i in range(5):
            layers.append(
                ConvTransposeBlock(
                        # in_channels=latent_dim if i == 0 else base_channels * 2 ** (3-i+1),
                        # out_channels=base_channels * 2 ** (3-i),
                        in_channels=in_channels if i == 0 else base_channels * 2 ** (5-i),
                        out_channels=base_channels * 2 ** (4-i),
                        kernel_size=4,
                        stride=1 if i == 0 else 2,
                        add_norm=True,
                        activation="ReLU"
                    )
                )
        layers.append(
            ConvTransposeBlock(
                    in_channels=base_channels,
                    out_channels=num_channels,
                    kernel_size=4,
                    stride=2,
                    add_norm=False,
                    activation="Tanh"
                )
            )
        
        self.model = nn.Sequential(*layers)
        return
    
    def forward(self, x, labels=None):
        """ Forward pass through generator 
        Args:
            x: latent vectors [B, latent_dim, 1, 1]
            labels: class labels [B] (optional, only used in conditional mode)
        """
        if labels is not None:
            # Conditional mode
            embedded_labels = self.label_embedding(labels)  # [B, num_classes]
            embedded_labels = embedded_labels.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1]
            x = torch.cat([x, embedded_labels], dim=1)
        
        # Generate image
        y = self.model(x)
        return y
    

class Discriminator(nn.Module):
    """ A flexible fully convolutional discriminator that can work in both conditional and unconditional modes.
    Conditional mode: (B, num_channels, 64, 64), (B) --> (B, 1, 1, 1)
    Unconditional mode: (B, num_channels, 64, 64) --> (B, 1, 1, 1)
    """
    def __init__(self, in_channels=3, out_dim=1, base_channels=64, dropout=0.3, num_classes=3, conditioned=False):
        """ Module initializer """
        super().__init__()  
        
        self.conditioned = conditioned

        if self.conditioned:
            # Embedding layer for class labels (used in conditional mode)
            self.label_embedding = nn.Embedding(num_classes, 64 * 64)
            self.num_classes = num_classes
        
        # Channel multipliers for each layer (instead of exponential growth---> less parameters in the model)
        channel_mults = [1, 2, 4, 4, 8]  
        
        layers = []
        in_channels = in_channels if not conditioned else in_channels + 1  # +1 for label channel

        # channel progression: 3 -> 64 -> 128 -> 256 -> 256 -> 512
        # spatial dims: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        for i in range(5):  
            layers.append(
                ConvBlock(
                    in_channels=in_channels if i == 0 else base_channels * channel_mults[i-1],
                    out_channels=base_channels * channel_mults[i],
                    kernel_size=4,
                    stride=2,
                    add_norm=True,
                    activation="LeakyReLU",
                    dropout=dropout
                )
            )
        
        # 2x2 -> 1x1. No padding, normal conv2d
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=base_channels * channel_mults[-1],  # 512 channels
                          out_channels=out_dim, 
                          kernel_size=2, 
                          stride=2, 
                          padding=0),
                nn.Sigmoid()
            )
        )
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, labels=None):
        """ Forward pass 
        Args:
            x: input images [B, C, H, W]
            labels: class labels [B] (optional, only used in conditional mode)
        """
        if labels is not None:
            # Conditional mode
            batch_size = x.shape[0]
            embedded_labels = self.label_embedding(labels)  # [B, H*W]
            embedded_labels = embedded_labels.view(batch_size, 1, 64, 64)  # [B, 1, H, W]
            x = torch.cat([x, embedded_labels], dim=1)
            
        y = self.model(x)
        return y
    
class Trainer:
    """
    Class for initializing GAN and training it in both conditional and unconditional modes
    """
    def __init__(self, generator, discriminator, latent_dim=128, num_classes=3, writer=None, conditioned=False):
        """ Initialzer """
        assert writer is not None, f"Tensorboard writer not set..."
    
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditioned = conditioned
        self.writer = writer 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.9))
        self.optim_generator = torch.optim.Adam(self.generator.parameters(), lr=3e-4, betas=(0.5, 0.9))
        
        # REAL LABEL = 1
        # FAKE LABEL = 0
        # eps = 1e-10
        # self.criterion_d_real = lambda pred: torch.clip(-torch.log(1 - pred + eps), min=-10).mean()
        # self.criterion_d_fake = lambda pred: torch.clip(-torch.log(pred + eps), min=-10).mean()
        # self.criterion_g = lambda pred: torch.clip(-torch.log(1 - pred + eps), min=-10).mean()
        
        self.criterion_g = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_real = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_fake = lambda pred: F.binary_cross_entropy(pred, torch.zeros(pred.shape[0], device=pred.device))
        
        
        self.hist = {
            "d_real": [],
            "d_fake": [],
            "g": []
        }
        return
        
    def train_one_step(self, imgs, labels=None):
        """ 
        Training both models for one optimization step
        Args:
            imgs: real images [B, C, H, W]
            labels: class labels [B] (optional, only for conditional mode)
        """
        self.generator.train()
        self.discriminator.train()
        
        # Sample from the latent distribution
        B = imgs.shape[0]
        latent = torch.randn(B, self.latent_dim, 1, 1).to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # ==== Training Discriminator ====
        self.optim_discriminator.zero_grad()
        # Get discriminator outputs for the real samples
        prediction_real = self.discriminator(imgs, labels)
        # Compute the loss function
        d_loss_real = self.criterion_d_real(prediction_real.view(B))

        # Generating fake samples with the generator
        fake_samples = self.generator(latent, labels)
        # Get discriminator outputs for the fake samples
        prediction_fake_d = self.discriminator(fake_samples.detach(), labels)  # why detach?
        # Compute the loss function
        d_loss_fake = self.criterion_d_fake(prediction_fake_d.view(B))
        (d_loss_real + d_loss_fake).backward()
        assert fake_samples.shape == imgs.shape
        
        # optimization step
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3.0)
        self.optim_discriminator.step()
        
        # === Train the generator ===
        self.optim_generator.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples, labels)
        # Compute the loss function
        g_loss = self.criterion_g(prediction_fake_g.view(B))
        g_loss.backward()
        # optimization step
        self.optim_generator.step()
        
        return d_loss_real, d_loss_fake, g_loss
    
    @torch.no_grad()
    def generate(self, N=64):
        """ Generating a bunch of images using current state of generator """
        self.generator.eval()
        latent = torch.randn(N, self.latent_dim, 1, 1).to(self.device)
        
        if self.conditioned:
            imgs_per_class = [21, 21, 22] # 64 images, 64 is not divisible by 3!
            num_classes = [0, 1, 2] # 3 classes

            labels = []
            for n, c in zip(imgs_per_class, num_classes):
                labels.append(torch.full((n,), c))  # creates [c, c, ..., c] of length n

            labels = torch.cat(labels).to(self.device)
            imgs = self.generator(latent, labels)
        else:
            # Generate images without conditioning
            imgs = self.generator(latent)
            
        imgs = imgs * 0.5 + 0.5
        return imgs
        
    def train(self, data_loader, N_iters=10000, init_step=0):
        """ Training the models for several iterations """
        
        progress_bar = tqdm(total=N_iters, initial=init_step)
        running_d_loss = 0
        running_g_loss = 0
        
        iter_ = 0
        for i in range(N_iters):
            for batch in data_loader:
                if self.conditioned:
                    real_batch, labels = batch
                    labels = labels.to(self.device)
                else:
                    real_batch = batch[0]  # Ignore labels if present
                    labels = None
                    
                real_batch = real_batch.to(self.device)
                d_loss_real, d_loss_fake, g_loss = self.train_one_step(imgs=real_batch, labels=labels)
                d_loss = d_loss_real + d_loss_fake
            
                # updating progress bar
                progress_bar.set_description(f"Ep {i+1} Iter {iter_}: D_Loss={round(d_loss.item(),5)}, G_Loss={round(g_loss.item(),5)})")
                
                # adding stuff to tensorboard
                self.writer.add_scalar(f'Loss/Generator Loss', g_loss.item(), global_step=iter_)
                self.writer.add_scalar(f'Loss/Discriminator Loss', d_loss.item(), global_step=iter_)
                self.writer.add_scalars(f'Loss/Discriminator Losses', {
                        "Real Images Loss": d_loss_real.item(),
                        "Fake Images Loss": d_loss_fake.item(),
                    }, global_step=iter_)
                self.writer.add_scalars(f'Comb_Loss/Losses', {
                            'Discriminator': d_loss.item(),
                            'Generator':  g_loss.item()
                        }, iter_)    
                if(iter_ % 200 == 0):
                    imgs = self.generate()
                    grid = torchvision.utils.make_grid(imgs, nrow=8)
                    self.writer.add_image('images', grid, global_step=iter_)
                    torchvision.utils.save_image(grid, os.path.join(os.getcwd(), "imgs", "training", f"imgs_{iter_}.png"))

                iter_ = iter_ + 1 
                
        return