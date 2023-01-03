import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from models.vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 42 # For reproducibility

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=128, type=int, help="batch size of examples")
    parser.add_argument("--download_mnist", default=True, type=bool, help="download MNIST")
    parser.add_argument("--model_ckpt", type=str, help="model checkpoint for evaluation")
    args = parser.parse_args()

    # Image transformer
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Download dataset
    test_dataset = MNIST('./data', transform=img_transform, download=True, train=False)

    # Dataset length
    num_test = len(test_dataset)
    print(f"Num. testing samples: {num_test}")

    # Data loader 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = VGG16.load(args.model_ckpt).to(device)

    print(">> Model's Architecture: ")
    print(best_model)
    print(f">> Total parameters: {count_parameters(best_model)}")

    criterion = nn.CrossEntropyLoss()
    
    best_model.eval()
    test_iterator = tqdm(test_loader, leave=True)
    running_test_loss, running_test_acc = 0.0, 0.0
    for tidx, (test_images, test_labels) in enumerate(test_iterator):
        with torch.no_grad():
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = best_model(test_images)
            test_loss = criterion(test_outputs, test_labels)
            test_acc = ((test_outputs.argmax(dim=1) == test_labels).float().mean())

            running_test_loss += test_loss.item()
            running_test_acc += test_acc.item()

            test_iterator.set_description('(Test)')
            test_iterator.set_postfix(test_loss=test_loss.item(), test_acc=test_acc.item())

    total_test_acc = running_test_acc/len(test_loader)
    total_test_loss = running_test_loss/len(test_loader)

    print(f'>> Result:\tTest loss = {total_test_loss:.5f} \t Test Acc = {total_test_acc:.5f}')

if __name__=="__main__":
    main()