import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
import os
import sys
from torchvision import transforms

curr_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from model import BreastCancerModel
from dataset_loader import CustomDataset

#Training configuration
BATCH_SIZE=16
LEARNING_RATE=0.0001
NUM_EPOCHS=10
base_dir=os.path.dirname(curr_dir)
data_path=os.path.join(base_dir,"data","image_data.csv")
save_path=os.path.join(base_dir,"saved_models","resnet50_mammogram.pth")

def get_device():
    # Device configuration (a graphics card if available)
    if torch.backends.mps.is_available():
        return torch.device("mps") #Mac M1/M2/M3
    elif torch.backends.cuda.is_available():
        return torch.device("cuda") #Nvidia GPU
    else:
        return torch.device("cpu") #Standard CPU


device=get_device()

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

def evaluate(model,loader,device):
    #validation
    model.eval()
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    
    return 100*correct/total


def train_model():
    full_train_dataset=CustomDataset(data_path,transform=data_transforms['train'])
    full_val_dataset=CustomDataset(data_path,transform=data_transforms['val'])
    # full_dataset=CustomDataset(data_path)

    #split the data- 80% for train and 20% for the validation
    dataset_size=len(full_train_dataset)
    indices=list(range(dataset_size))
    split=int(np.floor(0.2*dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset=Subset(full_train_dataset,train_indices)
    val_dataset=Subset(full_val_dataset,val_indices)

    #dataLoader
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True) #shuffle=True so that the network does not learn a fixed order
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

    #init the model
    model=BreastCancerModel().to(device)

    #define loss function and optimizer
    criterion=nn.CrossEntropyLoss() #normal/benign/malignant
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)

    #training loop
    for epoch in range(NUM_EPOCHS):
        model.train() #train mode
        running_loss=0.0
        correct=0
        total=0

        for images,labels in train_loader:
            #All calculations will be on the same device
            images=images.to(device)
            labels=labels.to(device)

            optimizer.zero_grad() # Reset the gradients from the previous iteration
            outputs=model(images)
            loss=criterion(outputs,labels) # Compares the output to the truth labels
            loss.backward() # Backpropagation- calculates the gradients of the weights according to the loss
            optimizer.step() #Updating the weights according to the calculated gradients

            running_loss+=loss.item()
            _,predicted=torch.max(outputs.data,1) #The most likely class for each sample
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    
        train_accuracy=100*correct/total
        avg_loss=running_loss/len(train_loader)

        val_acc=evaluate(model,val_loader,device)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        torch.save(model.state_dict(), save_path)
    

if __name__ == "__main__":
    train_model()

