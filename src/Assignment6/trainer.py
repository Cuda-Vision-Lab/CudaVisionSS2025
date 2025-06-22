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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

train_dataset = utils.TripletDataset(X_train, y_train, transform)
test_dataset = utils.TripletDataset(X_test, y_test, transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

margin = 1.0
temperature = 0.5
n_iters = 10000

model = models.SiameseModel()
criterion = utils.TripletLoss(margin=margin)

trainer = utils.Trainer(model=model, criterion=criterion, train_loader=train_loader, valid_loader=test_loader, n_iters=n_iters)

trainer.fit()

stats = {
    "train_loss": trainer.train_loss,
    "valid_loss": trainer.valid_loss,
    "margin": margin,
    "temperature": temperature
}
utils.save_model(trainer.model, trainer.optimizer, trainer.iter_, stats, margin=margin, temperature=temperature)

