import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats, margin, temperature):
    """ Saving model checkpoint """
    
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_epoch_{epoch}_margin_{margin}_temperature_{temperature}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def visualize_progress(train_loss, val_loss, start=0):
    """ Visualizing loss and accuracy """
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_train = smooth(train_loss, 31)
    ax[0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_yscale("linear")
    ax[0].set_title("Training Progress (linear)")
    
    ax[1].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (log)")

    smooth_val = smooth(val_loss, 31)
    N_ITERS = len(val_loss)
    ax[2].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[2].plot(np.arange(start, N_ITERS)+start, smooth_val[start:], c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("CE Loss")
    ax[2].set_yscale("log")
    ax[2].set_title(f"Valid Progress")

    return

def display_projections(points, labels, ax=None, legend=None):
    """ Displaying low-dimensional data projections """
    
    COLORS = ['r', 'b', 'g', 'y', 'purple', 'orange', 'k', 'brown', 'grey',
              'c', "gold", "fuchsia", "lime", "darkred", "tomato", "navy"]
    
    # If no legend provided, create default legend
    unique_labels = np.unique(labels)
    if legend is None:
        legend = [f"Class {l}" for l in unique_labels]
    
    if ax is None:
        _, ax = plt.subplots(1,1,figsize=(12,12))
    
    # Create a mapping from label values to indices
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    for i, l in enumerate(unique_labels):
        idx = np.where(labels == l)
        # Use the label's position in unique_labels to index into legend
        ax.scatter(points[idx, 0], points[idx, 1], label=legend[i], c=COLORS[i % len(COLORS)])
    ax.legend(loc="center left")


class NormLayer(nn.Module):
    """ Layer that computer embedding normalization """
    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2] # L1 or L2 normalization
        super().__init__()
        self.l = l
        return
    
    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized
    
class TripletLoss(nn.Module):
    """ Implementation of the triplet loss function """
    def __init__(self, margin=0.2, temperature = 1.0, reduce="mean"):
        """ Module initializer """
        assert reduce in ["mean", "sum"]
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.reduce = reduce
        return
        
    def forward(self, anchor, positive, negative):
        """ Computing temperature-scaled distances and loss """
        # Scale embeddings by temperature
        anchor = anchor / self.temperature
        positive = positive / self.temperature
        negative = negative / self.temperature
        
        # L2 distances
        d_ap = (anchor - positive).pow(2).sum(dim=-1)
        d_an = (anchor - negative).pow(2).sum(dim=-1)
        
        # triplet loss with temperature scaling
        loss = (d_ap - d_an + self.margin)
        loss = torch.maximum(loss, torch.zeros_like(loss))
        
        # averaging or summing      
        loss = torch.mean(loss) if(self.reduce == "mean") else torch.sum(loss)
      
        return loss
    
class Trainer:
    """
    Class for training and validating a siamese model
    """
    
    def __init__(self, model, criterion, train_loader, valid_loader, n_iters=1e4):
        """ Trainer initializer """
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.n_iters = int(n_iters)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.train_loss = []
        self.valid_loss = []
        return
    
    @torch.no_grad()
    def valid_step(self, val_iters=100):
        """ Some validation iterations """
        self.model.eval()
        cur_losses = []
        for i, ((anchors, positives, negatives),_) in enumerate(self.valid_loader):   
            # setting inputs to GPU
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)
            
            # forward pass and triplet loss
            anchor_emb, positive_emb, negative_emb = self.model(anchors, positives, negatives)
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            cur_losses.append(loss.item())
            
            if(i >= val_iters):
                break
    
        self.valid_loss += cur_losses
        self.model.train()
        
        return cur_losses
    
    def fit(self):
        """ Train/Validation loop """
    
        self.iter_ = 0
        progress_bar = tqdm(total=self.n_iters, initial=0)
        
        for i in range(self.n_iters):
            for (anchors, positives, negatives), _ in self.train_loader:     
                # setting inputs to GPU
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                
                # forward pass and triplet loss
                anchor_emb, positive_emb, negative_emb = self.model(anchors, positives, negatives)
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                self.train_loss.append(loss.item())
                
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                # updating progress bar
                progress_bar.set_description(f"Train Iter {self.iter_}: Loss={round(loss.item(),5)})")
                
                # doing some validation every once in a while
                if(self.iter_ % 250 == 0):
                    cur_losses = self.valid_step()
                    print(f"Valid loss @ iteration {self.iter_}: Loss={np.mean(cur_losses)}")
                
                self.iter_ = self.iter_+1 
                if(self.iter_ >= self.n_iters):
                    break
            if(self.iter_ >= self.n_iters):
                break
        return


def plot_PCA(imgs_flat, embs, labels, legend):
    """ Plotting PCA of images and embeddings """
    pca_imgs = PCA(n_components=2).fit_transform(imgs_flat)
    pca_embs = PCA(n_components=2).fit_transform(embs)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,2,figsize=(26,10))

    
    N = len(labels)  # Use all points
    display_projections(pca_imgs[:N], labels[:N], ax=ax[0], legend=legend)
    ax[0].set_title("PCA Proj. of Images")
    display_projections(pca_embs[:N], labels[:N], ax=ax[1], legend=legend)
    ax[1].set_title("PCA Proj. of Embeddings")
    plt.show()
    return


def plot_TSNE(imgs_flat, embs, labels, legend):
    N = 2000
    tsne_imgs = TSNE(n_components=2).fit_transform(imgs_flat[:N])
    tsne_embs = TSNE(n_components=2).fit_transform(embs[:N])

    fig,ax = plt.subplots(1,2,figsize=(26,10))
    display_projections(tsne_imgs[:N], labels[:N], ax=ax[0], legend=legend)
    ax[0].set_title("T-SNE Proj. of Images")
    display_projections(tsne_embs[:N], labels[:N], ax=ax[1], legend=legend)
    ax[1].set_title("T-SNE Proj. of Embeddings")
    plt.show()
    return tsne_imgs, tsne_embs


def calculate_ARI(imgs_flat, embs, labels):
    kmeans_imgs = KMeans(n_clusters=10, n_init=10,random_state=0).fit(imgs_flat)
    kmeans_embs = KMeans(n_clusters=10, n_init=10,random_state=0).fit(embs)    
    ari_imgs = adjusted_rand_score(labels, kmeans_imgs.labels_)
    ari_embs = adjusted_rand_score(labels, kmeans_embs.labels_)
    print(f"Clustering images achieves  ARI={round(ari_imgs*100,2)}%")
    print(f"Clustering embeddings achieves ARI={round(ari_embs*100,2)}%")

def display_projections_images(points, labels, dataset, ax=None, legend=None):
    """ Displaying low-dimensional data projections using images instead of points """

    COLORS = ['r', 'b', 'g', 'y', 'purple', 'orange', 'k', 'brown', 'grey',
              'c', "gold", "fuchsia", "lime", "darkred", "tomato", "navy",
              "pink", "cyan", "magenta", "olive", "teal", "coral", "indigo", 
              "violet", "maroon", "turquoise", "salmon", "plum", "sienna", 
              "orchid", "peru", "mediumseagreen", "lightcoral", "darkgreen", 
              "darkblue", "darkviolet", "darkorange", "darkcyan", "darkmagenta"]
    
    legend = [f"Class {l}" for l in np.unique(labels)] if legend is None else legend
    _, ax = plt.subplots(1,1,figsize=(36,24))
    
    # Plot points
    for i, l in enumerate(np.unique(labels)):
        idx = np.where(labels == l)
        ax.scatter(points[idx, 0], points[idx, 1], label=legend[i], color=COLORS[i % len(COLORS)])
    
    # Add image thumbnails
    for i, point in enumerate(points):
        xy = [point[0], point[1]]
        
        arr_img = dataset[i][0][0].permute(1, 2, 0).numpy()
        l = labels[i]
        imagebox = OffsetImage(arr_img, zoom=1)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0, 0),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            bboxprops=dict(edgecolor=COLORS[l], lw=2)
                            )

        ax.add_artist(ab)
    ax.legend(loc="best")
    plt.show()

class TripletDataset:
    """
    Dataset class from which we sample random triplets
    """
    def __init__(self, dataset, labels, transform=None):
        """ Dataset initializer"""
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        
        # Calculate dimensions for reshaping
        total_pixels = dataset.shape[1] // 3
        self.height = int(total_pixels ** 0.5)  # Assuming square images
        self.width = total_pixels // self.height
        return
    
    def __len__(self):
        """ Returning number of anchors """
        return len(self.labels)
    
    def __getitem__(self, i):
        """ 
        Sampling a triplet for the dataset. Index i corresponds to anchor 
        """
        # sampling anchor
        anchor_img = self.dataset[i].reshape(self.height, self.width, 3)  # Reshape to HxWx3
        anchor_lbl = self.labels[i]

        # lists for positives and negatives
        pos_ids = np.where(self.labels == anchor_lbl)[0]
        neg_id = np.where(self.labels != anchor_lbl)[0]
        
        # random positive and negative
        pos_id, neg_id = random.choice(pos_ids).item(), random.choice(neg_id).item()  # BIG FLAW HERE! 
        pos_img, pos_lbl = self.dataset[pos_id].reshape(self.height, self.width, 3) , self.labels[pos_id]
        neg_img, neg_lbl = self.dataset[neg_id].reshape(self.height, self.width, 3) , self.labels[neg_id]
              
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            
        return (anchor_img, pos_img, neg_img), (anchor_lbl, pos_lbl, neg_lbl)
    
def get_embeddings(model, test_loader, device):
    imgs_flat = []
    embs = []
    labels = []
    with torch.no_grad():
        for (anchor, _, _), (lbl,_, _) in test_loader:
            anchor = anchor.to(device)
            anchor_emb = model.forward_one(anchor)
            
            labels.append(lbl)
            embs.append(anchor_emb.cpu())
            imgs_flat.append(anchor.cpu().flatten(1))

    labels = np.concatenate(labels)
    embs = np.concatenate(embs)
    imgs_flat = np.concatenate(imgs_flat)
    return imgs_flat, embs, labels