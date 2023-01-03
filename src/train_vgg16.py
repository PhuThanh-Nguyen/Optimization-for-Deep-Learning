import os
import numpy as np
import glob
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models.vgg16 import VGG16
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from PIL import Image
from torch.utils.data import Subset
from optim.adam import AdamOptimizer
from optim.sgd import SgdOptimizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 42 # For reproducibility

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", default=50, type=int, help="max number training epoch")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size of examples")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--download_mnist", default=True, type=bool, help="download MNIST")
    parser.add_argument("--optim", default="adam", type=str, help="optim for training, ['adam', 'sgd']")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--nesterov", default=False, type=bool, help="only use for sgd optim")
    parser.add_argument("--output_dir", default="./output", type=str, help="output directory")    
    args = parser.parse_args()

    # Image transformer
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Download dataset
    train_dataset = MNIST('./data', transform=img_transform, download=True, train=True)

    # Dataset length
    num_train = len(train_dataset)
    print(f"Num. training samples: {num_train}")

    # Fraction of the original train set that we want to use as validation set
    val_frac = 0.2
    # Number of samples of the validation set
    num_val = int(num_train * val_frac) 
    num_train = num_train - num_val

    print(f"{num_train} samples used as train dataset")
    print(f"{num_val}  samples used as val dataset")

    # Split train_dataset into training and validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val], 
                                                            generator=torch.Generator().manual_seed(SEED))
    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16(args).to(device)

    print(">> Model's Architecture: ")
    print(model)
    print(f">> Total parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()

    if args.optim == 'adam':
        optim = AdamOptimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = SgdOptimizer(model.parameters(), lr=args.lr, 
                            weight_decay=args.weight_decay, 
                            momentum=args.momentum, nesterov=args.nesterov)

    # =================================================================
    history = {'acc': {'train': [], 'val': []}, 
    'loss': {'train': [], 'val': []}}
    min_val_loss = np.inf

    for epoch in range(args.num_epochs):
        # Training
        train_iterator = tqdm(train_loader, leave=True)
        running_train_loss = 0.0
        running_train_acc = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_iterator):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_acc = ((outputs.argmax(dim=1) == labels).float().mean())        
            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Gather training loss and acc
            running_train_loss += loss.item()
            running_train_acc += train_acc.item()
            
            train_iterator.set_description('(Train) Epoch [{}/{}]'.format(epoch, args.num_epochs))
            train_iterator.set_postfix(train_loss=loss.item(), train_acc=train_acc.item())

        epoch_train_loss = running_train_loss/len(train_loader)
        epoch_train_acc = running_train_acc/len(train_loader)

        history['acc']['train'].append(epoch_train_acc)
        history['loss']['train'].append(epoch_train_loss)
        
        # Evaluation
        val_iterator = tqdm(val_loader, leave=True)
        running_val_loss = 0.0
        running_val_acc = 0.0
        model.eval()
        for vidx, (val_images, val_labels) in enumerate(val_iterator):
            with torch.no_grad():
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_acc = ((val_outputs.argmax(dim=1) == val_labels).float().mean())

                running_val_loss += val_loss.item()
                running_val_acc += val_acc.item()
                
                val_iterator.set_description('(Val) Epoch [{}/{}]'.format(epoch, args.num_epochs))
                val_iterator.set_postfix(val_loss=val_loss.item(), val_acc=val_acc.item())
                
        epoch_val_acc = running_val_acc/len(val_loader)
        epoch_val_loss = running_val_loss/len(val_loader)

        history['acc']['val'].append(epoch_val_acc)
        history['loss']['val'].append(epoch_val_loss)
        
        print(f'>> Epoch [{epoch+1}/{args.num_epochs}]:\t(Train|Val) Loss = [{epoch_train_loss:.4f} | {epoch_val_loss:.4f}],\
                    \t (Train|Val) Accuracy = [{epoch_train_acc:.4f} | {epoch_val_acc:.4f}]')
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            print(">> Saving The Model Checkpoint")
            format_save = 'vgg16-{}-mnist.pt'.format(args.optim if not args.nesterov else 'nesterov')
            torch.save(
                {
                    'model_config': model.config,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'history': history
                }, os.path.join(args.output_dir, format_save)
            )

if __name__=="__main__":
    main()