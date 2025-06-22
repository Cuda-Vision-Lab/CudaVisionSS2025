import torch
import torch.nn as nn
from torchvision import models
import utils


class SiameseModel(nn.Module):
    """ 
    Implementation of a simple siamese model 
    """
    def __init__(self, emb_dim=64):
        """ Module initializer """
        super().__init__()

        # initialize resnet as backbone for feature extraction
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim)
        self.cnn = self.resnet

        # auxiliar layers
        # self.flatten = nn.Flatten()
        self.norm = utils.NormLayer()
    
        return
    
    def forward_one(self, x):
        """ Forwarding just one sample through the model """
        x_emb = self.cnn(x)
        # x_flat = self.flatten(x)
        # x_emb = self.fc(x)
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