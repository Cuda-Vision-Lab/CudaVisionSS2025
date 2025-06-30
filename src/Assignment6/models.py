import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import utils


class SiameseModel(nn.Module):
    """ 
    Implementation of a simple siamese model 
    """
    def __init__(self, emb_dim=64):
        """ Module initializer """
        super().__init__()

        # initialize resnet as backbone for feature extraction
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim)
        self.resnet.fc = nn.Identity()

        # projection head
        self.proj_head = nn.Sequential(
                nn.Linear(512, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )
        self.cnn = self.resnet

        # auxiliar layers
        # self.flatten = nn.Flatten()
        self.norm = utils.NormLayer()
    
        return
    
    def forward_one(self, x):
        """ Forwarding just one sample through the model """
        x_emb = self.proj_head(self.cnn(x))
        x_emb_norm = self.norm(x_emb)
        return x_emb_norm
    
    def forward(self, anchor, positive, negative):
        """ Forwarding a triplet """
        # anchor_emb = self.forward_one(anchor)
        # positive_emb = self.forward_one(positive)
        # negative_emb = self.forward_one(negative)

        # is there a more efficient way? anchor (B, C, H, W)
        all_inputs = torch.cat([anchor, positive, negative], dim=0)  # (3 * B, C, H, W)
        all_embs = self.forward_one(all_inputs)
        anchor_emb, positive_emb, negative_emb = all_embs.chunk(3, dim=0)
        
        return anchor_emb, positive_emb, negative_emb
    


class SimCLR(nn.Module):
        """
        SimCLR model with ResNet-18 Backbone
        """
        def __init__(self, hidden_dim=512, output_dim=128):
            super().__init__()

            # Use a pre-trained ResNet18 but modify for grayscale input
            self.backbone = models.resnet18(weights=None)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Identity()

            # projection head
            self.proj_head = nn.Sequential(
                    nn.Linear(512, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            return
        
        def forward(self, x):
            h = self.backbone(x)
            z = self.proj_head(h)
            out = F.normalize(z, dim=1)  # L2 normalize
            return out