from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import torch.nn as nn
import utils
# from imblearn.over_sampling import RandomOverSampler
import models
from tqdm import tqdm
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

lfw_people = fetch_lfw_people(min_faces_per_person=50, color=True, resize=1.0,
                              slice_=(slice(48, 202), slice(48, 202)))
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names


print(f"Dataset shape: {X.shape}")
print(f"Total images: {X.shape[0]}")
print(f"Image dimensions: {X.shape[1]} pixels")
print(f"Total labels: {len(y)}")
print(f"Number of people/classes: {len(target_names)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42) 

def train_siamese(model, margin, temperature, n_iters):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])

    train_dataset = utils.TripletDataset(X_train, y_train, transform)
    test_dataset = utils.TripletDataset(X_test, y_test, transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    criterion = utils.TripletLoss(margin=margin, temperature=temperature)

    trainer = utils.Trainer(model=model, criterion=criterion, train_loader=train_loader, valid_loader=test_loader, n_iters=n_iters)

    trainer.fit()

    stats = {
        "train_loss": trainer.train_loss,
        "valid_loss": trainer.valid_loss,
        "margin": margin,
        "temperature": temperature
    }
    utils.save_model(trainer.model, trainer.optimizer, trainer.iter_, stats, margin=margin, temperature=temperature)
    return

def train_simclr(model, temperature):
    from models import SimCLR
    NUM_EPOCHS = 300
    TEMP = temperature
    BATCH_SIZE = 64
    LR = 3e-3

    # data
    simclr_db = utils.SimCLRDataset(X_train, y_train)  
    data_loader = torch.utils.data.DataLoader(
            simclr_db,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0 
        )

    # model and optimizer
    model = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    model.train()
    losses = []

    print("Starting SimCLR training...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, (x_i, x_j,_) in enumerate(tqdm(data_loader)):
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            # Forward pass
            z_i = model(x_i)
            z_j = model(x_j)
            
            # Compute contrastive loss
            loss, _ = utils.nt_xent_loss(z_i, z_j, TEMP)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {avg_loss:.4f}')

    stats = {
        "losses": losses
            }
    utils.save_model(model, optimizer, epoch+1, stats,0, temperature)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="siamese")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n_iters", type=int, default=10000)
    args = parser.parse_args()

    if args.model == "siamese":
        model = models.SiameseModel()
        train_siamese(model, args.margin, args.temperature, args.n_iters)
    elif args.model == "simclr":
        model = models.SimCLR()
        train_simclr(model, args.temperature)
    else:
        raise ValueError(f"Invalid model: {args.model}")

if __name__ == "__main__":
    main()
