import os
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize_images(images):
    """Denormalize images from [-1, 1] to [0, 1] range"""
    return (images + 1) / 2

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def save_model(model, model_name, optimizer, epoch, stats):
    """ Saving model checkpoint """
    
    savepath = f"models/{model_name}/checkpoint_epoch_{epoch}.pth"
    if(not os.path.exists(savepath)):
        os.makedirs(savepath, exist_ok=True)

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



def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    recons_loss = []
    vae_loss = []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, _) in progress_bar:
        images = images.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass
        recons, (z, mu, log_var) = model(images)
         
        # Calculate Loss
        loss, (mse, kld) = criterion(recons, images, mu, log_var)
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
def eval_model(model, eval_loader, criterion, device, epoch=None, savefig=False, savepath="", writer=None):
    """ Evaluating the model for either validation or test """
    loss_list = []
    recons_loss = []
    kld_loss = []
    
    for i, (images, _) in enumerate(eval_loader):
        images = images.to(device)
        
        # Forward pass 
        recons, (z, mu, log_var) = model(images)
                 
        loss, (mse, kld) = criterion(recons, images, mu, log_var)
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
                
                # Add the comparison grid
                comparison = torch.cat([images_denorm, recons_denorm])
                grid_comparison = torchvision.utils.make_grid(comparison, nrow=6, padding=2, normalize=False)
                writer.add_image('Original vs Reconstructed', grid_comparison, epoch)
            
    # Total correct predictions and loss
    loss = np.mean(loss_list)
    recons_loss = np.mean(recons_loss)
    kld_loss = np.mean(kld_loss)
    return loss, recons_loss, kld_loss


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader,
                num_epochs, savepath, writer, save_frequency=5, vis_frequency=2):
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
                model=model, eval_loader=valid_loader, criterion=criterion,
                device=device, epoch=epoch, savefig=log_epoch, savepath=savepath,
                writer=writer
            )
        val_loss.append(loss)
        val_loss_recons.append(recons_loss)
        val_loss_kld.append(kld_loss)

        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)
        writer.add_scalars(f'Loss/All_Valid_Loss', {"recons": recons_loss.item(), "kld": kld_loss.item()}, global_step=epoch)
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            writer.add_scalar('Learning Rate', lr, global_step=epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
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
            save_model(model=model, optimizer=optimizer, epoch=epoch, stats=stats)
        
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
            # Denormalize images by multiplying by 0.5 and adding 0.5 (reverse of Normalize([0.5]*3, [0.5]*3))
            ax[0, i].imshow(imgs[i, 0] * 0.5 + 0.5)
            ax[0, i].axis("off")
            ax[1, i].imshow(recons[i, 0] * 0.5 + 0.5)
            ax[1, i].axis("off")

        ax[0, 3].set_title("Original Image")
        ax[1, 3].set_title("Reconstruction")
        plt.tight_layout()
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