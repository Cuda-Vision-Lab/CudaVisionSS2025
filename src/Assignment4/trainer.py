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
from torch.utils.data import DataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
from cvae import CVAE

def main(configs):
        
    dataset_root = './data/AFHQ/'

    transform = transforms.Compose([transforms.Resize((64,64)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5]*3 , [0.5]*3)])


    BS = configs["batch_size"]

    train_dataset = datasets.ImageFolder(root= dataset_root+'train', transform= transform )
    test_dataset = datasets.ImageFolder(root= dataset_root+'test', transform= transform )

    print(train_dataset.classes)  
    print(train_dataset.class_to_idx)  

    train_loader = DataLoader(dataset= train_dataset, 
                            batch_size= BS, 
                            shuffle= True, 
                            drop_last= True )

    test_loader = DataLoader(dataset= test_dataset, 
                            batch_size= BS, 
                            shuffle= False, 
                            drop_last= True )


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CVAE(latent_dim=configs["latent_dim"]).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs["lr"], weight_decay=1e-4)

    # Use a more gradual learning rate schedule
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, verbose=True)

    model_name = configs["model_name"]+configs["exp"]
    savepath = f"imgs/{model_name}"
    os.makedirs(savepath,exist_ok=True)

    TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", model_name)
    if not os.path.exists(TBOARD_LOGS):
        os.makedirs(TBOARD_LOGS)
    shutil.rmtree(TBOARD_LOGS)
    writer = SummaryWriter(TBOARD_LOGS)

    train_loss, val_loss, loss_iters, val_loss_recons, val_loss_kld = train_model(
            model=model, 
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler if configs["use_scheduler"] else None, 
            criterion=vae_loss_function,
            train_loader=train_loader, 
            valid_loader=test_loader, 
            num_epochs=configs["num_epochs"], 
            savepath=savepath,
            writer=writer
        )

if __name__ == "__main__":

    configs = {   
    "model_name" : "CVAE",
    "exp" : "3",  
    "latent_dim" : 256,
    "batch_size" : 64,
    "num_epochs" : 50,
    "lr" : 5e-4,
    "scheduler" : "ReduceLROnPlateau",
    "use_scheduler" : True,
    "lambda_kld" : 1e-3,
    }

    def vae_loss_function(recons, target, mu, log_var, lambda_kld = configs["lambda_kld"]):
            """
            Combined loss function for joint optimization of 
            reconstruction and ELBO
            """
            recons_loss = F.mse_loss(recons, target)
            kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)  # closed-form solution of KLD in Gaussian
            loss = recons_loss + lambda_kld * kld

            return loss, (recons_loss, kld)

    main(configs)