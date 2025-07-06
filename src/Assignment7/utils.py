# utils
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats, experiment_name):
    """ Saving model checkpoint """
    
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_{experiment_name}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def visualize_attention(image, attention_maps, patch_size=16, img_size=64):
    """ Overlaying the attention maps on the image """
    num_layers = len(attention_maps)
    num_heads, num_tokens = attention_maps[0].shape
    patches_per_side = img_size // patch_size
    num_patches = patches_per_side * patches_per_side
    
    # first displaying raw image
    fig, ax = plt.subplots(1, num_layers + 1)
    fig.set_size_inches(30, 5)
    ax[0].imshow(image, cmap='gray')
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    # displaying attention from each layer
    image_unnorm = (image * 255).astype(np.uint8)
    H, W = image.shape[:2]
    for i in range(num_layers):
        cur_attn = attention_maps[i][:, 1:]  # current attn and removing [CLS] token

        attn = cur_attn.mean(axis=0)  # average across heads 
        attn = attn / attn.max()  # renormalization
        attn_grid = attn.reshape(patches_per_side, patches_per_side)  # mapping back to image

        # Resize to image resolution        
        attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_CUBIC)
        # attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_NEAREST)

        # cmap = "jet"
        cmap = "coolwarm"
        
        im = ax[i+1].imshow(image, cmap='gray')
        ax[i+1].imshow(attn_up, cmap='gray', alpha=0.01, extent=(0, W, H, 0))
        cbar = plt.colorbar(ax[i+1].imshow(attn_up, cmap=cmap, alpha=0.8, extent=(0, W, H, 0), vmin=0, vmax=1), ax=ax[i+1], fraction=0.046, pad=0.04, cmap=cmap)
        cbar.set_label('Attention Intensity', fontsize=15)
        ax[i+1].axis('off')
        ax[i+1].set_title(f"Attention Layer {i+1}/{num_layers}", fontsize=20)
        
    plt.show()

def visualize_progress(loss_iters, train_loss, val_loss, valid_acc, start=0):
    """ Visualizing loss and accuracy """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)
    
    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")
    
    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs, train_loss, c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")
    
    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")
    
    plt.show()
    return


def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device):
    """ Evaluating the model for either validation or test """
    correct = 0
    total = 0
    loss_list = []
    
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass only to get logits/output
        outputs = model(images)
                 
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
            
        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len( torch.where(preds==labels)[0] )
        total += len(labels)
                 
    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    
    return accuracy, loss


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epochs, tboard=None, start_epoch=0):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []

    for epoch in range(num_epochs):
        print(f"Started Epoch {epoch+1}/{num_epochs}...")
        
        # validation epoch
        print("  --> Running valdiation epoch")
        model.eval()  # important for dropout and batch norms
        accuracy, loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device
            )
        valid_acc.append(accuracy)
        val_loss.append(loss)
        tboard.add_scalar(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
        tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
        
        # training epoch
        print("  --> Running train epoch")
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        scheduler.step()
        train_loss.append(mean_loss)
        tboard.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

        loss_iters = loss_iters + cur_loss_iters
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print(f"    Valid Accuracy: {accuracy}%")
        print("\n")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, valid_acc