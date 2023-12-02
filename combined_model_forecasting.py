#import libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from numpy import asarray
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
parser.add_argument('--dataset_path', type=str, default='dataset_forecasting_original.xlsx')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--pretrained', type=str2bool, default=True)
parser.add_argument('--lr', type=float, default=0.0001389)
parser.add_argument('--gamma', type=float, default=0.9138)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--normalize_images', type=str2bool, default=True)
parser.add_argument('--unfold_images', type=str2bool, default=True)
parser.add_argument('--shuffle_folds', type=str2bool, default=False)
parser.add_argument('--colormap', type=str2bool, default=False)
parser.add_argument('--transforms', type=str2bool, default=False)
parser.add_argument('--ghi1_before', type=str2bool, default=False)
parser.add_argument('--ghi1_actual', type=str2bool, default=False)
parser.add_argument('--image_model', type=str, default='vgg16')
parser.add_argument('--segmented', type=str2bool, default=False)
args = parser.parse_args()
args_dict = vars(args)

# path to save the processed images
processed_images_path = os.path.join('processed_images',f'processed_unfolded_before_{args.ghi1_before}_segmented_{args.segmented}_images_forecasting.npy')

def process_jp2_images(paths):

    X_images = []

    # Upload the images
    for path in tqdm(paths):
        # Read the image
        image = plt.imread(path)
        # Convert to uint8
        image = ((image / 65535.0) * 255).astype(np.uint8)
        # Unfold the image if it is required
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
    
    return X_images

def process_png_images(paths):
    
    X_images = []
    # Upload the images
    for path in tqdm(paths):
        # Read the image
        RGBAimg = Image.open(path)
        RGBAimg = asarray(RGBAimg)
        RGBimg = RGBAimg[:,:,0:3]
        # Convert to uint8
        image = (RGBimg * 255).astype(np.uint8)
        # Unfold the image if it is required
        if args.unfold_images:
            # Obtener dimensiones de la imagen
            height, width, _ = image.shape
            # Convertir la imagen a coordenadas polares
            center = (width // 2, height // 2)
            max_radius = min(width, height) // 2
            polar_image = cv2.warpPolar(image, (max_radius, 360), center, max_radius, cv2.WARP_POLAR_LINEAR)
            image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Convert to RGB or Colormap
        if args.colormap:
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        # Resize the image
        resized_image = cv2.resize(image, (224,224))
        # Transpose the image according to PyTorch standards (C, H, W)
        transposed_image = resized_image.transpose(2,0,1)
        X_images.append(transposed_image)
    
    return X_images
    
# Verify if there are no processed images file
if not os.path.isfile(processed_images_path):

    # Upload xlsx with the data
    if args.ghi1_before:
        df = pd.read_excel(args.dataset_path, engine='openpyxl', sheet_name="30after_30before")
    else:
        df = pd.read_excel(args.dataset_path, engine='openpyxl', sheet_name="30after")
    df = df.dropna()
    df = df.reset_index(drop=True)

    if args.segmented:
        paths = df.segmented_image_path
        X_images = process_png_images(paths)
    else:
        paths = df.image_path
        X_images = process_jp2_images(paths)

    # Convert to numpy array
    X_images_np = np.array(X_images)
    # Save the processed images
    np.save(processed_images_path, X_images_np)
    # Save the new dataframe
    df.to_excel(f"dataset_forecasting_before_{args.ghi1_before}.xlsx", index=False)

else:
    # Load the processed images
    X_images_np = np.load(processed_images_path)
    # Loas the new dataframe
    df = pd.read_excel(f"dataset_forecasting_before_{args.ghi1_before}.xlsx", engine='openpyxl')

# Define the target variable
y = df.ghi1_30.values

# Convert datetime.time to minutes
def time_to_minutes(time_val):
    # Convert the string to a datetime object
    formato = "%H:%M:%S"
    if type(time_val) == str:
        time_val = datetime.strptime(time_val, formato).time()
    # Convert the datetime object to minutes
    return time_val.hour * 60 + time_val.minute

if args.ghi1_before and args.ghi1_actual:
    X_numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity'], 'ghi1_30_before': df['30_ghi1'], 'ghi1_actual': df['ghi1']})
elif args.ghi1_before and not args.ghi1_actual:
    X_numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity'], 'ghi1_30_before': df['30_ghi1']})
elif not args.ghi1_before and args.ghi1_actual:
    X_numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity'], 'ghi1_actual': df['ghi1']})
else:
    X_numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity']})
X_numerical['hour'] = X_numerical['hour'].apply(time_to_minutes)

#Normalizing the data
scaler = MinMaxScaler()
X_numerical[X_numerical.columns] = scaler.fit_transform(X_numerical)
y = scaler.fit_transform(y.reshape(-1,1))

def normalize_image_dataset(images):
    """
    Normaliza un conjunto de datos de imágenes utilizando el valor mínimo y máximo de todo el conjunto.
    """
    # Encontrar el valor mínimo y máximo en todo el conjunto de datos
    min_val = np.min(images)
    max_val = np.max(images)

    # Normalizar el conjunto de datos
    normalized_images = (images - min_val) / (max_val - min_val)

    return normalized_images

if args.normalize_images:
    X_images_np = normalize_image_dataset(X_images_np)

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

X_images_tensor = torch.tensor(X_images_np, dtype=torch.float32)
X_numerical_tensor = torch.tensor(X_numerical.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_images_tensor.shape, X_numerical_tensor.shape, y_tensor.shape)

# Create a dataset object
dataset = TensorDataset(X_images_tensor, X_numerical_tensor, y_tensor)

# Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set random seed
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)

# Inicialize wandb
wandb.init(
    project='Solar Irradiation',
    config=args_dict,
    name=f'forecasting_before_{args.ghi1_before}_actual_{args.ghi1_actual}_sorted_{args.image_model}_segmented_{args.segmented}_unfolded_images',
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
    
# Modelo ResNet18 para las imágenes
class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        resnet = models.resnet18(pretrained=True) 
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Modelo ANN para datos numéricos
class NumericalModel(nn.Module):
    def __init__(self):
        super(NumericalModel, self).__init__()
        if (args.ghi1_before and not args.ghi1_actual) or (not args.ghi1_before and args.ghi1_actual):
            self.fc = nn.Sequential(
                nn.Linear(4, 128),  
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        elif args.ghi1_before and args.ghi1_actual:
            self.fc = nn.Sequential(
                nn.Linear(5, 128),  
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3, 128),  
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
if args.image_model == 'vgg16':
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

elif args.image_model == 'resnet18':

    class CombinedModel(nn.Module):
        def __init__(self):
            super(CombinedModel, self).__init__()
            self.resnet = ResNetFeatures().to(device)
            self.numerical = NumericalModel().to(device)
            # Ajustar el tamaño de la entrada según las características de ResNet18
            self.fc = nn.Sequential(
                nn.Linear(512 + 64, 256),  # Ajuste basado en las características de ResNet18
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, image, numerical_data):
            image_features = self.resnet(image)
            numerical_features = self.numerical(numerical_data)
            combined = torch.cat((image_features, numerical_features), dim=1)
            output = self.fc(combined)
            return output

def calculate_metrics(preds, labels):
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    variance = np.var(labels - preds)
    return mse, rmse, mae, variance
    
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

            all_preds = []
            all_labels = []
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for images_inputs, numerical_inputs, labels in tqdm(dataloaders[phase]):
                images_inputs = images_inputs.to(device)
                numerical_inputs = numerical_inputs.to(device)
                labels = labels.to(device).squeeze()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images_inputs, numerical_inputs).squeeze()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images_inputs.size(0)
                # Guardar las predicciones y etiquetas para calcular métricas
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            if phase == 'train':
                scheduler.step()

            # Calcular loss de la época
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            wandb.log({f'{phase}_loss_fold{fold}': epoch_loss, 'epoch': epoch})
            
            # Calcular métricas
            mse, rmse, mae, variance = calculate_metrics(np.array(all_preds), np.array(all_labels))
            print(f"{phase.capitalize()} MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Variance: {variance:.4f}")
            wandb.log({f'{phase}_mse_fold{fold}': mse, f'{phase}_rmse_fold{fold}': rmse, 
                       f'{phase}_mae_fold{fold}': mae, f'{phase}_variance_fold{fold}': variance, 'epoch': epoch})

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                model_path = f'checkpoints/{wandb.run.name}.pth'
                torch.save(model.state_dict(), model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation loss: {:.4f}".format(best_loss))

    return mse, rmse, mae, variance

# Cross validation funtion
def cross_validation(dataset, k_folds, num_epochs, device):
    kfold = KFold(n_splits=k_folds, shuffle=args.shuffle_folds)
    results = {'mse': [], 'rmse': [], 'mae': [], 'variance': []}
    
    # Split dataset into K folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('-' * 10)
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)
        
        dataloaders = {"train": train_loader, "val": val_loader}
        mse, rmse, mae, variance = train_model(dataloaders, num_epochs=num_epochs, device=device, fold=fold)
        results['mse'].append(mse)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['variance'].append(variance)
    
    # Print average results of K-Fold validation
    wandb.log({'avg_mse':np.mean(results['mse']), 'avg_rmse': np.mean(results['rmse']), 'avg_mae': np.mean(results['mae']), 'avg_variance': np.mean(results['variance'])})
    
    return results

val_results = cross_validation(dataset=dataset, k_folds=args.k_folds, num_epochs=args.epochs, device=device)

wandb.finish()

