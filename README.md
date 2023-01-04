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

## MNIST Reconstruction result

<p align="center"> <img src="Miscellaneous/animation.gif" width=600 height=300> </p>
<p align="center"> <i> The training process of reconstruction problem on MNIST dataset using Nesterov SGD </i> </p>
<table border="0" align="center">
  <tr>
    <td> <img src="Miscellaneous/first epoch.png" width=600 height=300> </td>
    <td> <img src="Miscellaneous/last epoch.png" width=600 height=300> </td>
  </tr>
</table>
<p align="center"> <i> Reconstructed image at the first epoch (Left) and at the last epoch (Right) </i> </p>

## AutoEncoder
### Model's architecture
|  Layer  |       |  Size | Activation |
|:-------:|:-----:|:-----:|:----------:|
|  Input  | Image | 28x28 |      -     |
| Encoder |   FC  |  128  |    relu    |
|         |   FC  |   64  |    relu    |
|         |   FC  |   12  |    relu    |
|         |   FC  |   3   |    relu    |
| Decoder |   FC  |   3   |    relu    |
|         |   FC  |   12  |    relu    |
|         |   FC  |   64  |    relu    |
|         |   FC  |  128  |    tanh    |
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

| Layer |                 | Feature Map |      Size     | Kernel Size | Stride | Activation |
|:-----:|:---------------:|:-----------:|:-------------:|:-----------:|:------:|:----------:|
| Input |      Image      |      1      |  28 x 28 x 1  |      -      |    -   |      -     |
|   1   | 2 x Convolution |      64     |  28 x 28 x 64 |     3x3     |    1   |    relu    |
|       |   Max Pooling   |      64     |  14 x 14 x 64 |     2x2     |    2   |    relu    |
|   2   | 2 x Convolution |     128     | 14 x 14 x 128 |     3x3     |    1   |    relu    |
|       |   Max Pooling   |     128     |  7 x 7 x 128  |     2x2     |    2   |    relu    |
|   3   | 2 x Convolution |     256     |  7 x 7 x 256  |     3x3     |    1   |    relu    |
|       |   Max Pooling   |     256     |  3 x 3 x 256  |     2x2     |    2   |    relu    |
|   4   | 2 x Convolution |     512     |  3 x 3 x 512  |     3x3     |    1   |    relu    |
|       |   Max Pooling   |     512     |  1 x 1 x 512  |     2x2     |    2   |    relu    |
|   5   |   FC + Dropout  |      -      |      512      |      -      |    -   |    relu    |
|   6   |   FC + Dropout  |      -      |      256      |      -      |    -   |    relu    |
|   7   |   FC + Dropout  |      -      |      128      |      -      |    -   |    relu    |
|   8   |     SoftMax     |      -      |       10      |      -      |    -   |   softmax  |
### MNIST Classification result

| Method       | Accuracy | Settings                                  |
|--------------|----------|-------------------------------------------|
| SGD          | 0.99104  | lr=1e-3; momentum=0; weight_decay=0       |
| SGD-Nesterov | 0.99373  | lr=1e-3; momentum=0.86; weight_decay=1e-5 |
| Adam         | 0.99134  | lr=1e-3; weight_decay=1e-4                |


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
