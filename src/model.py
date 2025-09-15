import torch
import torch.nn as nn
import torch.nn.functional as F #Activation functions

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()

        #Convolution block 1
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1) #16 filters in the first layer, each filter is 3x3 in size
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2) #Halve the height and width

        #Convolution block2
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)

        #Fully connected layers
        
        #Input image size:224x224 -> After 2x maxpool(2,2): 56x56
        #output of conv2 block = [batch_size,32,56,56]-> so input to fc1= 32x56x56 
         
        self.fc1=nn.Linear(32*56*56,64)
        self.fc2=nn.Linear(64,3) #normal,benign,malignant

    def forward(self,x):
        # Conv → ReLU → Pool
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))

        #Flatten (batch size,featurs)
        x=x.view(x.size(0),-1)
        #fc1 → relu → fc2
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x



