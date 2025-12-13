import torch
import torch.nn as nn
from torchvision import models

class BreastCancerModel(nn.Module):
    def __init__(self,num_classes=3):
        super(BreastCancerModel,self).__init__()

        self.resnet=models.resnet50(weights='IMAGENET1K_V1') #IMAGENET1K_V1- Loads trained weights    
        
        # #Adjusting the entrance to the first layer (only the in channel to 1)
        # self.resnet.conv1=nn.Conv2d(
        #     in_channels=1,
        #     out_channels=64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False
        # )

        #Adjusting the fc layer to 3
        num_features=self.resnet.fc.in_features
        self.resnet.fc=nn.Linear(num_features,num_classes)
    
    def forward(self,x):
        return self.resnet(x)

