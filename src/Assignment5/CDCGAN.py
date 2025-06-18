#!/usr/bin/env python

import os
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import *
import models
# from models import Generator, Discriminator, Trainer ---> imports statically, not compatibel with %load_ext autoreload
from torch.utils.tensorboard import SummaryWriter


configs = {   
    "model_name" : "CDCGAN",
    "exp" : "2",  
    "latent_dim" : 128,
    "batch_size" : 64,
    "num_epochs" : 15,
    "lr" : 1e-3,
    "scheduler" : "ReduceLROnPlateau",
    "use_scheduler" : True,
    }

dataset_root = '../Assignment4/data/AFHQ/'

transform = transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5]*3 , [0.5]*3)])

BS = configs["batch_size"]
latent_dim = configs["latent_dim"]

train_dataset = datasets.ImageFolder(root= dataset_root+'train', transform= transform )
test_dataset = datasets.ImageFolder(root= dataset_root+'test', transform= transform )

# print(train_dataset.classes)  
print(train_dataset.class_to_idx)  

train_loader = DataLoader(dataset= train_dataset, 
                          batch_size= BS, 
                          shuffle= True, 
                          drop_last= True )

test_loader = DataLoader(dataset= test_dataset, 
                          batch_size= BS, 
                          shuffle= False, 
                          drop_last= True )


generator = models.Generator(latent_dim=latent_dim, num_channels=3, base_channels=64, num_classes=3, conditioned=True)
# print(generator)


labels = torch.randint(0, 3, (BS,))
gen_input = torch.rand(BS, 128, 1, 1)
gen_img = generator(gen_input, labels)
print(f'output shape: {gen_img.shape}')

assert gen_img.shape == (BS, 3, 64, 64), "Generator output shape is incorrect! The Generator should output a fake image equal to the size of the training images"


discriminator = models.Discriminator(in_channels=3, out_dim=1, base_channels=64, num_classes=3, conditioned=True)
# print(discriminator)

labels = torch.randint(0, 3, (BS,))
desc_input = torch.rand(BS, 3, 64, 64)
desc_output = discriminator(desc_input, labels)
print(f'output shape: {desc_output.shape}')
assert desc_output.shape == (BS, 1, 1, 1), "Discriminator output shape is incorrect! The Discriminator should output a single value"


count_model_params(discriminator)

count_model_params(generator)

seed_everything(42)

model_name = configs["model_name"]+configs["exp"]
savepath, writer = makedires(configs)

model = models.Trainer(generator=generator, 
                       discriminator=discriminator, 
                       latent_dim=latent_dim, 
                       writer=writer,
                       model_name=model_name,
                       conditioned=True)

epoch = configs["num_epochs"]

model.train(data_loader=train_loader)

save_model(model, model_name, model.optim_generator, model.optim_discriminator, epoch = epoch, stats = configs )
save_config(configs)

