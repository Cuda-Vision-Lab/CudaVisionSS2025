import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from Custom_Cell import ConvLSTMCell;


class MotionDataset:
    
    """
    Dataset for motion classification
    """
    
    LABELS = {
        0: "down",
        1: "up",
        2: "right",
        3: "left"
    }
    
    def __init__(self, train, transform, img_size=(64, 64), num_frames=5, resize=1, move=2):
        """ """ 
        self.img_size = img_size
        self.num_frames = num_frames
        self.resize = resize
        self.move = move
        self.mnist_db = datasets.MNIST(root='./data', train=train, transform=transform, download=True)
        return

    def __len__(self):
        """ """
        return len(self.mnist_db)
    
    def __getitem__(self, i):
        """ Creating a random sequence of a moving digit"""
        # sampling digit
        digit = self.mnist_db[i][0]
        digit = F.interpolate(digit.unsqueeze(0), scale_factor=1/self.resize)[0]
        digitH, digitW = digit.shape[-2], digit.shape[-1]
        
        # creating canvas
        canvasH, canvasW = self.img_size
        canvas = torch.zeros(self.num_frames, 1, canvasH, canvasW)
        start_pos_x = np.random.randint(3, canvasH - 3 - digitH)
        start_pos_y = np.random.randint(3, canvasH - 3 - digitH)
        start_pos = torch.tensor([start_pos_y, start_pos_x])
        
        # moving parameters
        n = np.random.rand()
        
        if n < 0.25:
            move = torch.tensor([self.move, 0])
            label = 0
        elif n < 0.5:
            move = torch.tensor([-self.move, 0])
            label = 1
        elif n < 0.75:
            move = torch.tensor([0, self.move])
            label = 2
        else:
            move = torch.tensor([0, -self.move])
            label = 3
        
        # updating positions and creating video
        positions = []
        for i in range(self.num_frames):
            if i == 0:
                cur_pos = start_pos
            else:
                cur_pos = positions[-1] + move
                if cur_pos[0] < 0 or cur_pos[0] >= canvasH - digitH or \
                   cur_pos[1] < 0 or cur_pos[1] >= canvasW - digitW:  # avoid going out of bounds
                    cur_pos = positions[-1]
            positions.append(cur_pos)
            canvas[i, :, cur_pos[0]:cur_pos[0]+digitH, cur_pos[1]:cur_pos[1] + digitW] = digit
        
        return canvas, label
    
def visualize_sequence(sequence,  suptitle="", add_title=True, add_axis=False, n_cols=5, size=1.5,  vmax=1, vmin=0, **kwargs):
    """ Visualizing a grid with several images/frames """
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(size * n_cols, size + 0.5)
    fig.suptitle(suptitle)
 
    ims = []
    fs = []
    for i in range(n_frames):
        a = ax[i]
        f = sequence[i, 0].cpu().detach()
        a.imshow(f, vmin=vmin, vmax=vmax, **kwargs)
        if add_title:
            a.set_title(f"Frame {i}", fontsize=12)
        if not add_axis:
            a.set_yticks([])
            a.set_xticks([])
    plt.tight_layout()
    return

train_dataset = MotionDataset(
        train=True,
        transform=transforms.ToTensor(),
        img_size=(32, 32),
        num_frames=4,
        resize=2
    )

test_dataset = MotionDataset(
        train=False,
        transform=transforms.ToTensor(),
        img_size=(32, 32),
        num_frames=4,
        resize=2
    )

frames, label = train_dataset[0]
print(frames.shape)


class LSTMWithCustomCell(nn.Module):
    """ 
    Sequential classifier. Embedded images are fed to a RNN
    Same as above, but using LSTMCells instead of the LSTM object
    
    Args:
    -----
    emb_dim: integer 
        dimensionality of the vectors fed to the LSTM
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        number of stacked LSTMS
    mode: string
        intialization of the states
    """
    
    def __init__(self, emb_dim, hidden_dim, num_layers=1, mode="zeros"):
        """ Module initializer """
        assert mode in ["zeros", "random"]
        super().__init__()
        self.hidden_dim =  hidden_dim
        self.num_layers = num_layers
        self.mode = mode

        # for embedding rows into vector representations
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, emb_dim, 3, 1, 1),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        # LSTM model       
        lstms = []
        for i in range(num_layers):
            in_size = emb_dim if i == 0 else hidden_dim
            #lstms.append( nn.LSTMCell(input_size=in_size, hidden_size=hidden_dim) )
            lstms.append( ConvLSTMCell(input_size=in_size, hidden_size=hidden_dim) )
            
        self.lstm = nn.ModuleList(lstms)
        
        # FC-classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=4)
        
        return
    
    
    def forward(self, x):
        """ Forward pass through model """
        
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape
        h, c = self.init_state(b_size=b_size, device=x.device) 
        
        # embedding rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(b_size, num_frames, -1)
        
        # iterating over sequence length
        lstm_out = []
        for i in range(embeddings.shape[1]):  # iterate over time steps
            lstm_input = embeddings[:, i, :]  # size= (batch_size, emb_dim) 
            # iterating over LSTM Cells
            for j, lstm_cell in enumerate(self.lstm):
                #try:
                    if lstm_input.shape[0] != B_SIZE:
                        continue
                    #print(lstm_input.shape)
                    h[j], c[j] = lstm_cell(lstm_input, (h[j], c[j]))
                    lstm_input = h[j]
                #except:
                    #lstm_input=lstm_input;
            lstm_out.append(lstm_input)
        lstm_out = torch.stack(lstm_out, dim=1)
            
        # classifying
        y = self.classifier(lstm_out[:, -1, :])  # feeding only output at last layer
        
        return y
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        elif(self.mode == "random"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        return h, c

for i in range(4):
    frames, label = train_dataset[i]
    visualize_sequence(frames, n_cols=4, cmap="gray", suptitle=f"Class={label} ({train_dataset.LABELS[label]})")

# Fitting data loaders for iterating
B_SIZE = 256

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=B_SIZE, 
                                           shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=B_SIZE,
                                          shuffle=False)
mmmm=1;
nnnn=1
def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        mmmm=i;
        nnnn=(images, labels);
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
        
        progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")
        
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
def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epochs):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []
    
    for epoch in range(num_epochs):
           
        # validation epoch
        model.eval()  # important for dropout and batch norms
        accuracy, loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device
            )
        valid_acc.append(accuracy)
        val_loss.append(loss)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        scheduler.step()
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print(f"    Accuracy: {accuracy}%")
        print("\n")
        """save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                stats={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "loss_iters": loss_iters,
                    "valid_acc": valid_acc,
                }
            )"""
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, valid_acc

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMWithCustomCell(emb_dim=128, hidden_dim=128, num_layers=2, mode="zeros")
count_model_params(model)
print(device, model.encoder)
count_model_params(model.encoder)
model = model.to(device)

# classification loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Decay LR by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
train_loss, val_loss, loss_iters, valid_acc = train_model(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loader=train_loader, valid_loader=test_loader, num_epochs=10
    )
