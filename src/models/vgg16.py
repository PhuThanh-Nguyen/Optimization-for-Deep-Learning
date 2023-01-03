import os
import torch
import torch.nn.functional as F
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        conv2_params = {
            'kernel_size': (3, 3),
            'stride'     : (1, 1),
            'padding'    : 1
        }

        self._batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **conv2_params)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else lambda x : x

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, **conv2_params)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else lambda x : x

        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    @property
    def batch_norm(self):
        return self._batch_norm

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.max_pooling(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, config):
        super(VGG16, self).__init__()
        self.config = config
        self.in_channels, self.in_width, self.in_height = config.input_size

        self.block_1 = VGGBlock(self.in_channels, 64, batch_norm=config.batch_norm)
        self.block_2 = VGGBlock(64, 128,batch_norm=config.batch_norm)
        self.block_3 = VGGBlock(128, 256,batch_norm=config.batch_norm)
        self.block_4 = VGGBlock(256, 512,batch_norm=config.batch_norm)

        self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Dropout(p=0.1),
                nn.Linear(128, config.num_classes) 
            )

    @property
    def input_size(self):
        return self.in_channels, self.in_width, self.in_height

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
    @staticmethod
    def _init_model_with_state_dict(state):
        model = VGG16(state['model_config'])
        model.load_state_dict(state['model_state_dict'])
        return model
    
    @classmethod
    def load(cls, path):
        r"""
        Loads a model with data fields and pretrained model parameters.
        Args:
            path (str):
                - a string with the shortcut name of a pretrained model
                  to load from .pt file.
        Examples:
            >>> # model = VGG16.load('./<model_name>.pt')
        """
        if os.path.exists(path):
            state = torch.load(path, map_location=torch.device('cpu'))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        model = cls._init_model_with_state_dict(state)
        return model