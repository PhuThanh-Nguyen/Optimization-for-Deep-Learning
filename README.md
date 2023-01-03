# Optimization for Deep Learning

This repository contains demonstrating code for numerical results in the report of the project `Optimization for Deep Learning` topic in Mathematical Modeling and Computational Optimization for Data Science coursework - French-Vietnamese applied mathematics master program 2022-2023.

Members for this group project:
<table align="center">
  <tr>
    <th>Name</th>
    <th>Student's ID</th>
  </tr>
  <tr>
    <td>Nguyen Phu Thanh</td>
    <td>22C30015</td>
  </tr>
  <tr>
    <td>Le Hoang Duc</td>
    <td>22C30003</td>
  </tr>
  <tr>
    <td>Le Ngoc Hoang Huan</td>
    <td>22C30005</td>
  </tr>
</table> 

## AutoEncoder

### Results

<p align="center"> <img src="Miscellaneous/animation.gif" width=600 height=300> </p>
<p align="center"> <i> The training process of reconstruction problem on MNIST dataset using Nesterov SGD </i> </p>
<table border="0" align="center">
  <tr>
    <td> <img src="Miscellaneous/first epoch.png" width=600 height=300> </td>
    <td> <img src="Miscellaneous/last epoch.png" width=600 height=300> </td>
  </tr>
</table>
<p align="center"> <i> Reconstructed image at the first epoch (Left) and at the last epoch (Right) </i> </p>

### To retrain AutoEncoder model for the reconstruction problem, run the following command:
```
python src/run_ae.py --num_epochs=20\
                     --batch_size=128\
                     --lr=1e-2\
                     --download_mnist=False\
                     --optim='sgd'\
                     --momentum=0.96\
                     --weight_decay=1e-4\
                     --nesterov=True\
                     --output_dir='output'\
                     --save_train_img=False\
```

## MNIST Classification with VGG16
### Model's architecture
```
VGG16(
  (block_1): VGGBlock(
    (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (max_pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (block_2): VGGBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (max_pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (block_3): VGGBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (max_pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (block_4): VGGBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (max_pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=512, out_features=256, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.65, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.65, inplace=False)
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

### Results

| Method       | Accuracy | Settings                                  |
|--------------|----------|-------------------------------------------|
| SGD          | 0.99184  | lr=1e-3; momentum=0; weight_decay=0       |
| SGD-Nesterov | 0.99303  | lr=1e-3; momentum=0.86; weight_decay=1e-5 |
| Adam         | 0.99363  | lr=1e-3; momentum=0; weight_decay=1e-4    |

### To retrain VGG-16 for the classification problem, run the following command:
```
python src/train_vgg16.py --num_epochs=50\ 
                          --batch_size=64\
                          --lr=1e-3\
                          --download_mnist=True\
                          --optim='adam'\
                          --output_dir='checkpoints'\
```
### To only run VGG-16 on the test dataset for the classification problem, run the following command:
```
python src/test_vgg16.py --batch_size=64\
                         --lr=1e-3\
                         --download_mnist=False\
                         --model_ckpt='checkpoints/vgg16-adam-mnist.pt'\
```