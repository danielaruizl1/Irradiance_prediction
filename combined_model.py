#import libraries
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import wandb
import torch
import time
import copy
import cv2
import os

# Add an argument parser for hyperparameters
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser(description='Code for solar irradiation prediction')
parser.add_argument('--dataset_path', type=str, default='dataset_original.xlsx')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--pretrained', type=str2bool, default=True)
parser.add_argument('--lr', type=float, default=0.0001389)
parser.add_argument('--gamma', type=float, default=0.9138)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--unfold_images', type=str2bool, default=False)
parser.add_argument('--colormap', type=str2bool, default=False)
parser.add_argument('--transforms', type=str2bool, default=False)
args = parser.parse_args()
args_dict = vars(args)

torch.manual_seed(2021)

processed_images_path = os.path.join('processed_images',f'processed_unfolded_{args.unfold_images}_colormap_{args.colormap}_images_uint8.npy')

# Verify if there are no processed images file
if not os.path.isfile(processed_images_path):

    # Upload xlsx with the data
    df = pd.read_excel(args.dataset_path, engine='openpyxl')
    df = df.dropna()
    df = df.reset_index(drop=True)

    X_images = []
    not_founded = []

    # Upload the images
    for path in tqdm(df.image_path):
        try:
            # Read the image
            image = plt.imread(path)
            # Convert to uint8
            image = ((image / 65535.0) * 255).astype(np.uint8)
            if args.unfold_images:
                # Obtener dimensiones de la imagen
                height, width = image.shape
                # Convertir la imagen a coordenadas polares
                center = (width // 2, height // 2)
                max_radius = min(width, height) // 2
                polar_image = cv2.warpPolar(image, (max_radius, 360), center, max_radius, cv2.WARP_POLAR_LINEAR)
                image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Convert to RGB or Colormap
            if args.colormap:
                color_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            else:
                color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Resize the image
            resized_image = cv2.resize(color_image, (224,224))
            # Transpose the image according to PyTorch standards (C, H, W)
            transposed_image = resized_image.transpose(2,0,1)
            X_images.append(transposed_image)
        except:
            print(f'{path} not found')
            not_founded.append(path)
            df = df[df['image_path'] != path]
 
    # Convert to numpy array
    X_images_np = np.array(X_images)
    # Save the processed images
    np.save(processed_images_path, X_images_np)
    # Save the new dataframe
    df.to_excel("dataset.xlsx", index=False)

else:
    # Load the processed images
    X_images_np = np.load(processed_images_path)
    # Loas the new dataframe
    df = pd.read_excel("dataset.xlsx", engine='openpyxl')

# Define the target variable
y = df.ghi1.values

# Convert datetime.time to minutes
def time_to_minutes(time_val):
    # Convert the string to a datetime object
    formato = "%H:%M:%S"
    if type(time_val) == str:
        time_val = datetime.strptime(time_val, formato).time()
    # Convert the datetime object to minutes
    return time_val.hour * 60 + time_val.minute

X_numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity']})
X_numerical['hour'] = X_numerical['hour'].apply(time_to_minutes)

# Define images transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations to the images
if args.transforms:
    for i in range(len(X_images_np)):
        X_images_np[i] = transform(X_images_np[i].transpose(1, 2, 0))

X_images_tensor = torch.tensor(X_images_np.astype(np.int16), dtype=torch.float32)
#X_images_tensor = torch.tensor(X_images_np, dtype=torch.float32)
X_numerical_tensor = torch.tensor(X_numerical.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_images_tensor.shape, X_numerical_tensor.shape, y_tensor.shape)

# Create a dataset object
dataset = TensorDataset(X_images_tensor, X_numerical_tensor, y_tensor)

# Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Inicialize wandb
wandb.init(
    project='Solar Irradiation',
    config=args_dict
)

# Modelo VGG16 para las imágenes
class VGG16Features(nn.Module):
    def __init__(self):
        super(VGG16Features, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Modelo ANN para datos numéricos
class NumericalModel(nn.Module):
    def __init__(self):
        super(NumericalModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
# Modelo combinado
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.vgg16 = VGG16Features().to(device)
        self.numerical = NumericalModel().to(device)
        self.fc = nn.Sequential(
            nn.Linear(25088 + 64, 256),  # Ajusta el tamaño según tus modelos
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image, numerical_data):
        image_features = self.vgg16(image)
        numerical_features = self.numerical(numerical_data)
        combined = torch.cat((image_features, numerical_features), dim=1)
        output = self.fc(combined)
        return output
    
def train_model(dataloaders, num_epochs, device, fold):
    """
    Trains a model and returns the best model.
    
    Args:
    - model: The neural network model.
    - criterion: The loss function.
    - optimizer: The optimization algorithm.
    - scheduler: Learning rate scheduler.
    - dataloaders: A dictionary containing training and validation dataloaders.
    - num_epochs: Number of training epochs.
    - device: The device to train on.
    - fold: The fold number.

    Returns:
    - The best model based on validation accuracy.
    """

    # Crear el modelo
    model = CombinedModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    since = time.time()
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

            for images_inputs, numerical_inputs, labels in tqdm(dataloaders[phase]):
                images_inputs = images_inputs.to(device)
                numerical_inputs = numerical_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images_inputs, numerical_inputs).squeeze()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images_inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            wandb.log({f'{phase}_loss_fold{fold}': epoch_loss, 'epoch': epoch})

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                model_path = f'checkpoints/{wandb.run.name}.pth'
                torch.save(model.state_dict(), model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation loss: {:.4f}".format(best_loss))

    return best_loss

# Cross validation funtion
def cross_validation(dataset, k_folds, num_epochs, device):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = []
    
    # Split dataset into K folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('-' * 10)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)
        
        dataloaders = {"train": train_loader, "val": val_loader}
        best_fold_loss = train_model(dataloaders, num_epochs=num_epochs, device=device, fold=fold)
        wandb.log({'fold': fold, 'best_val_loss': best_fold_loss})
        results.append(best_fold_loss)
    
    # Print average results of K-Fold validation
    print(f'Best Fold Loss: {np.min(results):.4f}')
    print(f'Average Val Loss: {np.mean(results):.4f}')
    print(f'Standard Deviation of Val Loss: {np.std(results):.4f}')
    wandb.log({'best_val_loss':np.min(results), 'avg_val_loss': np.mean(results), 'std_val_loss': np.std(results)})
    
    return results

val_results = cross_validation(dataset=dataset, k_folds=args.k_folds, num_epochs=args.epochs, device=device)

wandb.finish()

