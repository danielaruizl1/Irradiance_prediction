#%%
#import libraries
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import time
import copy
import cv2

#%%

tranform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Upload xlsx with the data
file_path = 'sensors-20230822.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df = df.dropna()
df = df.reset_index(drop=True)

# Prep the train, validation and test dataset
torch.manual_seed(2021)

X = []
y = df.ghi1.values

for path in df.image_path:
    image = plt.imread(path)
    image = ((image / 65535.0) * 255).astype(np.uint8)
    min_val = image.min()
    max_val = image.max()
    image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb_image = rgb_image.transpose(2,0,1)
    X.append(rgb_image)

X = np.array(X)

#%%
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE=1

# Divide datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  
y_train = torch.tensor(y_train, dtype=torch.float32)  
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)    
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create datasets 
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)

# train, val and test datasets to the dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#%%
# Plot some train set images
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(images.shape)
#images_normalized = (images - images.min()) / (images.max() - images.min())
#images_with_channel = images_normalized.unsqueeze(1)  
grid = make_grid(images, nrow=8) 
plt.figure(figsize=(15, 15))  
plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
plt.axis('off') 
plt.show()
plt.savefig("train_loader.png")
print(labels.shape)
print(labels)

#%%

# Declare the model and its parameters
model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, 1)
model_ft = model_ft.to(device)
criterion = nn.MSELoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)

#%%

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, device='cuda'):
    """
    Trains a model and returns the best model.
    
    Args:
    - model: The neural network model.
    - criterion: The loss function.
    - optimizer: The optimization algorithm.
    - scheduler: Learning rate scheduler.
    - dataloaders: A dictionary containing training and validation dataloaders.
    - num_epochs: Number of training epochs (default=25).
    - device: The device to train on (default='cuda').

    Returns:
    - The best model based on validation accuracy.
    """
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    #_, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation loss: {:.4f}".format(best_loss))

    model.load_state_dict(best_model_wts)
    return model

dataloaders = {"train":train_loader, "val":val_loader}
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=50, device=device)

# Define a file path to save the model to
model_path = 'checkpoints/best_model.pth'

# Save the model to the specified file path
torch.save(model_ft.state_dict(), model_path)
