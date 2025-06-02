import os
import shutil
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize_images(images):
    """Denormalize images from [-1, 1] to [0, 1] range"""
    return (images + 1) / 2

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def save_model(model, model_name, optimizer, epoch, lambda_kld, stats):
    """ Saving model checkpoint """
    
    # Create the directory path first
    save_dir = f"models/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Then create the full file path
    savepath = os.path.join(save_dir, f"checkpoint_KLD_{lambda_kld}_epoch_{epoch}.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats if stats is not None else None
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats



def train_epoch(model, train_loader, optimizer, criterion, lambda_kld, epoch, device, constrained=False):
    """ Training a model for one epoch """
    
    loss_list = []
    recons_loss = []
    vae_loss = []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass
        if not constrained:
            recons, (z, mu, log_var) = model(images)
        else:
            recons, (z, mu, log_var) = model(images, labels)
         
        # Calculate Loss
        loss, (mse, kld) = criterion(recons, images, mu, log_var, lambda_kld)
        loss_list.append(loss.item())
        recons_loss.append(mse.item())
        vae_loss.append(kld.item())
        
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")
        
    mean_loss = np.mean(loss_list)
    
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, lambda_kld, device, epoch=None, savefig=False, savepath="", writer=None, constrained=False):
    """ Evaluating the model for either validation or test """
    loss_list = []
    recons_loss = []
    kld_loss = []
    
    for i, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass 
        if not constrained:
            recons, (z, mu, log_var) = model(images)
        else:
            recons, (z, mu, log_var) = model(images, labels)
                 
        loss, (mse, kld) = criterion(recons, images, mu, log_var, lambda_kld)
        loss_list.append(loss.item())
        recons_loss.append(mse.item())
        kld_loss.append(kld.item())
        
        if(i==0 and savefig):
            # Denormalize images before saving
            recons_denorm = denormalize_images(recons[:36].cpu())
            images_denorm = denormalize_images(images[:36].cpu())
            
            # Save reconstructions
            save_image(recons_denorm, os.path.join(savepath, f"recons_{epoch}.png"), nrow=6, padding=2, normalize=False)
            # Save original images for comparison
            save_image(images_denorm, os.path.join(savepath, f"original_{epoch}.png"), nrow=6, padding=2, normalize=False)
            
            if writer is not None:
                # Add images to tensorboard
                grid_orig = torchvision.utils.make_grid(images_denorm, nrow=6, padding=2, normalize=False)
                grid_recon = torchvision.utils.make_grid(recons_denorm, nrow=6, padding=2, normalize=False)
                writer.add_image('Original Images', grid_orig, epoch)
                writer.add_image('Reconstructed Images', grid_recon, epoch)
            
    # Total correct predictions and loss
    loss = np.mean(loss_list)
    recons_loss = np.mean(recons_loss)
    kld_loss = np.mean(kld_loss)
    return loss, recons_loss, kld_loss


def train_model(model, model_name, optimizer, scheduler, criterion, lambda_kld, train_loader, valid_loader,
                num_epochs, savepath, writer, save_frequency=5, vis_frequency=2, constrained=False):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    val_loss_recons =  []
    val_loss_kld =  []
    loss_iters = []
    
    for epoch in range(num_epochs):
           
        # validation epoch
        model.eval()  # important for dropout and batch norms
        log_epoch = (epoch % vis_frequency == 0 or epoch == num_epochs - 1)
        loss, recons_loss, kld_loss = eval_model(
                model=model, eval_loader=valid_loader, criterion=criterion, lambda_kld=lambda_kld,
                device=device, epoch=epoch, savefig=log_epoch, savepath=savepath,
                writer=writer, constrained=constrained
            )
        val_loss.append(loss)
        val_loss_recons.append(recons_loss)
        val_loss_kld.append(kld_loss)

        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)
        writer.add_scalars(f'Loss/All_Valid_Loss', {"recons": recons_loss.item(), "kld": kld_loss.item()}, global_step=epoch)
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            writer.add_scalar('Learning Rate', lr, global_step=epoch)
        
        writer.add_scalar(f'Lambda/KLD', lambda_kld, global_step=epoch)

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, lambda_kld=lambda_kld, epoch=epoch, device=device, constrained=constrained
            )
        writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch)
        writer.add_scalars(f'Loss/Comb', {"train": mean_loss.item(), "valid": loss.item()}, global_step=epoch)
        
        # PLATEAU SCHEDULER
        scheduler.step(val_loss[-1])
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        
        if(epoch % save_frequency == 0):
            stats = {
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "loss_iters": loss_iters
            }
            # save_model(model=model, model_name=model_name, optimizer=optimizer, epoch=epoch, stats=stats)
        
        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
            print(f"       Valid loss recons: {round(val_loss_recons[-1], 5)}")
            print(f"       Valid loss KL-D:   {round(val_loss_kld[-1], 5)}")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, val_loss_recons, val_loss_kld

def img_vs_recons(model, test_loader, device):

    imgs, _ = next(iter(test_loader)) 
    model.eval()
    with torch.no_grad():
        recons, _ = model(imgs.to(device))
        fig, ax = plt.subplots(2, 8)
        fig.set_size_inches(18, 5)
        for i in range(8):
            # Denormalize images from [-1, 1] to [0, 1] range
            orig_img = denormalize_images(imgs[i].cpu())
            recon_img = denormalize_images(recons[i].cpu())
            
            # Convert from (C,H,W) to (H,W,C) format for imshow
            orig_img = orig_img.permute(1, 2, 0)
            recon_img = recon_img.permute(1, 2, 0)
            
            ax[0, i].imshow(orig_img)
            ax[0, i].axis("off")
            ax[1, i].imshow(recon_img)
            ax[1, i].axis("off")

        ax[0, 3].set_title("Original Image")
        ax[1, 3].set_title("Reconstruction")
        plt.tight_layout()
        plt.show()


def plot_recons(recons):
    plt.figure(figsize=(8*2, 4*2))
    for i in range(32):
        plt.subplot(4,8,i+1)
        # recon_img = denormalize_images(recons[i].cpu())
        recon_img = recons[i].cpu()
        recon_img = recon_img.permute(1, 2, 0)
        plt.imshow(recon_img)  
        plt.axis("off")
    plt.tight_layout()
    plt.show()

COLORS = ['r', 'b', 'g', 'y', 'purple', 'orange', 'k', 'brown', 'grey',
          'c', "gold", "fuchsia", "lime", "darkred", "tomato", "navy"]

def display_projections(points, labels, ax=None, legend=None):
    """ Displaying low-dimensional data projections """
    
    legend = [f"Class {l}" for l in np.unique(labels)] if legend is None else legend
    if(ax is None):
        _, ax = plt.subplots(1,1,figsize=(12,6))
    
    for i,l in enumerate(np.unique(labels)):
        idx = np.where(l==labels)

        ax.scatter(points[idx, 0], points[idx, 1], label=legend[int(l)], c=COLORS[i])
    ax.legend(loc="best")

# @torch.no_grad()
# def plot_reconstructed(model, xrange=(-3, 3), yrange=(-2, 2), N=12):
#     """
#     Sampling equispaced points from the latent space given the xrange and yrange, 
#     decoding latents and visualizing distribution of the space
#     """
#     # Project points to decoder input dimension (same as in forward pass)
#     SIZE = 64  # Image size
#     grid = np.empty((N*SIZE, N*SIZE, 3))  # 3 channels for RGB
    
#     for i, y in enumerate(np.linspace(*yrange, N)):
#         for j, x in enumerate(np.linspace(*xrange, N)):
#             # mean
#             mu = torch.zeros(model.latent_dim, device=device)
#             mu[0] = x
#             mu[1] = y
            
#             # standard deviation
#             sigma = 1 
#             z = torch.normal(mean=mu, std=sigma)
            
#             # Passing through the decoder 
#             z = model.decoder_input(z)
#             z = z.view(-1, 256, 4, 4)  # Reshape to match decoder input
            
#             # Getting recons
#             x_hat = model.decoder(z)
            
#             # To visualize
#             x_hat = x_hat.squeeze(0).cpu()  # Remove batch dimension
#             x_hat = x_hat.permute(1, 2, 0)  # (C,H,W) to (H,W,C)
#             x_hat = x_hat.numpy()
            
#             # Enhance contrast
#             x_hat = np.clip(x_hat, 0, 1)  # Multiply by 1.2 to increase contrast, then clip to valid range
            
#             grid[(N-1-i)*SIZE:(N-i)*SIZE, j*SIZE:(j+1)*SIZE] = x_hat
           
#     plt.figure(figsize=(12,20))
#     plt.imshow(grid, extent=[*yrange, *xrange])
#     plt.axis("off")
#     plt.show()

@torch.no_grad()
def plot_reconstructed(model, xrange=(-3, 3), yrange=(-2, 2), N=12):
    """
    Sampling equispaced points from the latent space given the xrange and yrange, 
    decoding latents and visualizing distribution of the space
    """
    # Project points to decoder input dimension (same as in forward pass)
    SIZE = 64  # Image size
    grid = np.empty((N*SIZE, N*SIZE, 3))  # 3 channels for RGB
    
    for i, y in enumerate(np.linspace(*yrange, N)):
        for j, x in enumerate(np.linspace(*xrange, N)):
            # mean
            mu = torch.zeros(model.latent_dim, device=device)
            mu[0] = x
            mu[1] = y
            
            # standard deviation
            sigma = 1 
            std = torch.full_like(mu, sigma)
            z = torch.normal(mean=mu, std=std)
            
            # Passing through the decoder 
            z = model.decoder_input(z)
            z = z.view(-1, 256, 4, 4)  # Reshape to match decoder input
            
            # Getting recons
            x_hat = model.decoder(z)
            
            # To visualize
            x_hat = x_hat.squeeze(0).cpu()  # Remove batch dimension
            x_hat = x_hat.permute(1, 2, 0)  # (C,H,W) to (H,W,C)
            recon_img = x_hat.numpy()
            
            # Prepare the image for visualization
            recon_img = denormalize_images(recon_img[i])
            # recon_img = recon_img.permute(1, 2, 0)
            
            grid[(N-1-i)*SIZE:(N-i)*SIZE, j*SIZE:(j+1)*SIZE] = recon_img
           
    plt.figure(figsize=(12,20))
    plt.imshow(grid, extent=[*yrange, *xrange])
    plt.axis("off")
    plt.show()

def vae_loss_function(recons, target, mu, log_var, lambda_kld=0.0):
    """
    Combined loss function for joint optimization of 
    reconstruction and ELBO
    """
    recons_loss = F.mse_loss(recons, target)
    kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)  # closed-form solution of KLD in Gaussian
    loss = recons_loss + lambda_kld * kld

    return loss, (recons_loss, kld)

def makedires(configs):
    model_name = configs["model_name"]+configs["exp"]+f"_KLD_{configs['lambda_kld']}"
    savepath = f"imgs/{model_name}"
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath,exist_ok=True)

    TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", model_name)
    if os.path.exists(TBOARD_LOGS):
        shutil.rmtree(TBOARD_LOGS)
    os.makedirs(TBOARD_LOGS)
    writer = SummaryWriter(TBOARD_LOGS)
    return savepath, writer

def save_config(configs):
    model_name = configs["model_name"]+configs["exp"]+f"_KLD_{configs['lambda_kld']}"
    configs_dir = f"./configs/{model_name}/"
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir,exist_ok=True)
    configs_path = configs_dir + "/config.yaml"

    with open(configs_path, 'w') as f:
        yaml.dump(configs, f)


def compute_encoder_output_size(batch_size, input_shape, model):
    """
    Compute the un/flattened output size of the encoder given an input shape.
    """
    with torch.no_grad():

        enc_input = torch.zeros(batch_size, *input_shape).to(device)  # (BS, 3, 64, 64)
        
        # Get encoder output
        output = model.encoder(enc_input)
        
        enc_out_shape = output.view(batch_size,-1,4,4).shape[1:]
        # Get flattened size
        flattened_size = output.view(1, -1).shape[1]
        
    return enc_out_shape, flattened_size

def inference(configs, model):
    model_name = configs["model_name"]+configs["exp"]+f"_KLD_{configs['lambda_kld']}"
    BS = configs["batch_size"]
    if not os.path.exists(f"imgs/inference/{model_name}"):
        os.makedirs(f"imgs/inference/{model_name}")

    latent_dim = configs["latent_dim"]

    enc_output_shape, _ = compute_encoder_output_size(BS, (3, 64, 64), model)
    # print(enc_output_shape)
    with torch.no_grad():
        for i in range(5):
            z = torch.randn(BS, latent_dim).to(device)

            z = model.decoder_input(z)
            z = z.view(-1, *enc_output_shape)

            recons = model.decoder(z)
            recons = recons.view(BS, 3, 64, 64)
            save_image(recons, f"imgs/inference/{model_name}/inference_{i}.png")
            return recons

def plot_recons(recons):
    plt.figure(figsize=(8*2, 4*2))
    for i in range(32):
        plt.subplot(4,8,i+1)
        recon_img = denormalize_images(recons[i].cpu())
        # recon_img = recons[i].cpu()
        recon_img = recon_img.permute(1, 2, 0)
        plt.imshow(recon_img)  
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    

def vis_latent(test_loader, model, test_dataset):
    imgs_flat, latents, labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            _, (z, _, _) = model(imgs)
            imgs_flat.append(imgs.cpu().view(imgs.shape[0],-1))
            latents.append(z.cpu())
            labels.append(lbls)
            
    imgs_flat = np.concatenate(imgs_flat)    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)

    latents_reshaped = latents.reshape(latents.shape[0], -1)

    pca_imgs = PCA(n_components=2).fit_transform(imgs_flat)
    pca_latents = PCA(n_components=2).fit_transform(latents_reshaped)

    N = 2000
    fig,ax = plt.subplots(1,2,figsize=(26,8))
    display_projections(pca_imgs[:N], labels[:N], ax=ax[0], legend=test_dataset.classes)
    ax[0].set_title("PCA Proj. of Images")
    display_projections(pca_latents[:N], labels[:N], ax=ax[1], legend=test_dataset.classes)
    ax[1].set_title("Encoded Representations")
    plt.show()


def compute_stats(dataset, channels = 3):
    """Computing mean and std of dataset"""
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    num_samples = 0

    for img, _ in tqdm(dataset):  # img shape: [3, H, W]
        mean += img.mean(dim=(1, 2))  # Per-channel mean
        std += img.std(dim=(1, 2))    # Per-channel std
        num_samples += 1

    mean /= num_samples
    std /= num_samples
    return mean, std

def get_activation(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()

def get_dropout(drop_p):
    """ Getting a dropout layer """
    if(drop_p):
        drop = nn.Dropout(p=drop_p)
    else:
        drop = nn.Identity()
    return drop
def reparameterize(mu, log_var):
    """ Reparametrization trick"""
    std = torch.exp(0.5*log_var)  # we can also predict the std directly, but this works best
    eps = torch.randn_like(std)  # random sampling happens here
    z = mu + std * eps
    return z
