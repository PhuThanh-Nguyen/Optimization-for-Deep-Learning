"""
    Train AutoEncoder
"""

import os 
import torch
import torch.nn as nn
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.autograd import Variable
from models.autoencoder import AutoEncoder
from optim.adam import AdamOptimizer
from optim.sgd import SgdOptimizer
from tqdm import tqdm
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

SEED = 42 # For reproducibility

# ========= helper ===========
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

# def show_img(x):
#     grid_img = torchvision.utils.make_grid(x, nrow = 16)
#     plt.imshow(grid_img.permute(1, 2, 0))
#     plt.show()

def show_img(x, epoch, save_dir=None):
    grid_img = torchvision.utils.make_grid(x, nrow = 16)
    fig, ax = plt.subplots(figsize = (10, 15))
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.set(xticks = [], yticks = [])
    plt.title(f'Epoch = {epoch}')
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'{epoch}.png'), bbox_inches='tight')
    plt.show()
# =============================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", default=50, type=int, help="max number training epoch")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size of examples")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--download_mnist", default=True, type=bool, help="download MNIST")
    parser.add_argument("--optim", default="adam", type=str, help="optim for training, ['adam', 'sgd']")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.0, type=float, help="momentum")
    parser.add_argument("--nesterov", default=False, type=bool, help="only use for sgd optim")
    parser.add_argument("--output_dir", default="./output", type=str, help="output directory")
    parser.add_argument("--save_train_img", default=False, type=bool, help="save reconstruct training image")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Image transformer
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Download dataset
    train_dataset = MNIST('./data', transform=img_transform, download=True, train=True)
    val_dataset = MNIST('./data', transform=img_transform, download=True, train=False)

    # Dataset length
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    print(f"Num. training samples:   {num_train}")
    print(f"Num. validated samples:  {num_val}")

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()

    if args.optim == "adam":
        optim = AdamOptimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = SgdOptimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)


    # Loop
    history = {'train': [], 'val': []}

    for epoch in range(args.num_epochs):
        recon_loss = 0.0
        train_iterator = tqdm(train_loader, leave=True)
        model.train()
        for data in train_iterator:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optim.zero_grad()
            loss.backward()
            optim.step()
            recon_loss += loss.item()

            # ================== log ========================
            train_iterator.set_description('(Train) Epoch [{}/{}]'.format(epoch, args.num_epochs))
            train_iterator.set_postfix(train_recon_loss=loss.item())
        
        recon_loss /= len(train_loader)
        history['train'].append(recon_loss)

        if args.save_train_img:
            if epoch % 10 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, os.path.join(args.output_dir, 'image_{}.png'.format(epoch)))

        # =================== Validation ==================
        # clear_output(wait=True)
        model.eval()
        val_iterator = tqdm(val_loader, leave=True)
        val_recon_loss = 0.0
        for eval_batch in val_iterator:
            val_img, _ = eval_batch
            val_img = val_img.view(val_img.size(0), -1)
            val_img = Variable(val_img).to(device)
            with torch.no_grad():
                # ===================forward=====================
                val_output = model(val_img)
                val_loss = criterion(val_output, val_img)
                val_recon_loss += val_loss.item()
                
                # ================== log ========================
                val_iterator.set_description('(Val) Epoch [{}/{}]'.format(epoch, args.num_epochs))
                val_iterator.set_postfix(val_recon_loss=val_loss.item())

                # ================== save result ==================
                val_orig_pic = to_img(val_img.cpu().data)
                val_recon_pic = to_img(val_output.cpu().data)
                # show_img(valpic, epoch, args.output_dir)

                save_image(val_orig_pic, os.path.join(args.output_dir, 'val_orig_{}.png'.format(epoch)))
                save_image(val_recon_pic, os.path.join(args.output_dir, 'val_recon_{}.png'.format(epoch)))

        val_recon_loss /= len(val_loader)
        history['val'].append(val_recon_loss)
        print(f"Epoch [{epoch}/{args.num_epochs}]: recons loss (train | val) = ({recon_loss:.4f} | {val_recon_loss:.4f})\n")

    # Save model checkpoint
    if args.nesterov:
        format_save = 'ae-{}.pt'.format(args.optim + '-nesterov')
    else:
        format_save = 'ae-{}.pt'.format(args.optim)
    torch.save({
        'model_config': args,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'history': history
    }, os.path.join(args.output_dir, format_save))

if __name__=="__main__":
    main()
