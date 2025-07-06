from utils import *
import shutil
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import yaml
from transformations import *
from dataloader import KTHActionDataset
from utils import count_model_params
from torch.utils.tensorboard import SummaryWriter
from models import TransformerBlock, ViT

categories = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']

# configs = {   
#     "model_name" : "ViT",
#     "batch_size" : 32,
#     "num_epochs" : 60,
#     "lr" : 3e-4, # 1e-4
#     "scheduler" : "StepLR",
#     "use_scheduler" : False,
#     'max_frames' : 80,
#     'max_len' : 100,
#     'slicing_step' : 8,
#     'patch_size' : 8,
#     'token_dim' : 128,
#     'attn_dim' : 128,
#     'num_heads' : 4,
#     'mlp_size' : 512,
#     'num_tf_layers' : 4,
#     'num_classes' : len(categories)
# }

configs = {   
    "model_name" : "ViT",
    "batch_size" : 32,
    "num_epochs" : 100,
    "lr" : 3e-4, # 1e-4
    "scheduler" : "StepLR",
    "use_scheduler" : False,
    'max_frames' : 80,
    'max_len' : 100,
    'slicing_step' : 8,
    'patch_size' : 16,
    'token_dim' : 192,
    'attn_dim' : 192,
    'num_heads' : 4,
    'mlp_size' : 768,
    'num_tf_layers' : 6,
    'num_classes' : len(categories)
}
# Save configs to YAML file
cfg_dir = os.path.join(os.getcwd(), "configs")
os.makedirs(cfg_dir, exist_ok=True)
cfg_name = configs['model_name']+f"_patch_size_{configs['patch_size']}_epochs_{configs['num_epochs']}"
with open(os.path.join(cfg_dir, f'{cfg_name}.yaml'), 'w') as file:
    yaml.dump(configs, file, default_flow_style=False, sort_keys=False)

# Create datasets
root_dir = "/home/nfs/inf6/data/datasets/kth_actions/processed/"

train_dataset = KTHActionDataset(root_dir, 
                                 split="train", 
                                 transform=get_train_transforms(configs['slicing_step']), 
                                 max_frames=configs['max_frames'], 
                                 img_size=(64, 64))

test_dataset = KTHActionDataset(root_dir, 
                                split="test", 
                                transform=get_test_transforms(configs['slicing_step']), 
                                max_frames=configs['max_frames'], 
                                img_size=(64, 64))


# Create DataLoaders
train_loader = DataLoader(train_dataset, 
                          batch_size=configs['batch_size'], 
                          shuffle=True, 
                          num_workers=4)

test_loader = DataLoader(test_dataset, 
                         batch_size=configs['batch_size'], 
                         shuffle=False, 
                         num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(42) # Don't forget to seed!

tf_block = TransformerBlock(
        token_dim=configs['token_dim'],
        attn_dim=configs['attn_dim'],
        num_heads=configs['num_heads'],
        mlp_size=configs['mlp_size']
    ).to(device)

print(f"Transformer-Block has {count_model_params(tf_block)} parameters")


vit = ViT(
        patch_size=configs['patch_size'],
        token_dim=configs['token_dim'],
        attn_dim=configs['attn_dim'],
        num_heads=configs['num_heads'],
        mlp_size=configs['mlp_size'],
        num_tf_layers=configs['num_tf_layers'],
        num_classes=configs['num_classes'],
        C=1, # number of channels in the image
        max_len=configs['max_len']
    ).to(device)

print(f"ViT has {count_model_params(vit)} parameters")

# classification loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(vit.parameters(), lr=configs['lr'])

# Decay LR by a factor of 3 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1/3)

TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", configs['model_name']+f"_patch_size_{configs['patch_size']}_epochs_{configs['num_epochs']}")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)

shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)


train_loss, val_loss, loss_iters, valid_acc = train_model(
        model=vit,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=test_loader,
        num_epochs=configs['num_epochs'],
        tboard=writer
    )

stats = {
    "train_loss": train_loss,
    "valid_loss": val_loss,
    "loss_iters": loss_iters,
    "valid_acc": valid_acc
}
experiment_name = f"{configs['model_name']}_patch_size_{configs['patch_size']}_epochs_{configs['num_epochs']}"
save_model(vit, optimizer, epoch=configs['num_epochs'], stats=stats, experiment_name=experiment_name)

