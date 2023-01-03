# Optimization for Deep Learning

<p align="center"> <img src="Miscellaneous/animation.gif" width=600 height=300> </p>
<p align="center"> <i> The training process of reconstruction problem on MNIST dataset using Nesterov SGD </i> </p>
<table border="0" align="center">
  <tr>
    <td> <img src="Miscellaneous/first epoch.png" width=600 height=300> </td>
    <td> <img src="Miscellaneous/last epoch.png" width=600 height=300> </td>
  </tr>
</table>
<p align="center"> <i> Reconstructed image at the first epoch (Left) and at the last epoch (Right) </i> </p>

Demonstrating code for numerical results in the report of the project 'Optimization for Deep Learning'.

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


### To retrain AutoEncoder model for the reconstruction problem, run the following command:
```
python src/run_ae.py --num_epochs=20\
                     --batch_size=128\
                     --lr=1e-2\
                     --download_mnist=False\
                     --optim='adam'\
                     --weight_decay=0.96\
                     --nesterov=False\
                     --output_dir='output'\
                     --save_train_img=False\
```
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
