import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import os
import sys

curr_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from model import BreastCancerModel
from dataset_loader import CustomDataset

#Training configuration
BATCH_SIZE=16
LEARNING_RATE=0.001
NUM_EPOCHS=10

def train_model():
    # Device configuration (a graphics card if available)
    if torch.backends.mps.is_available():
        device=torch.device("mps")
    elif torch.backends.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    base_dir=os.path.dirname(curr_dir)
    data_path=os.path.join(base_dir,"data","image_data.csv")
    save_path=os.path.join(base_dir,"saved_models","resnet50_mammogram.pth")

    full_dataset=CustomDataset(data_path)

    #split the data- 80% for train and 20% for the validation
    train_size=int(0.8*len(full_dataset))
    val_size=len(full_dataset)-train_size
    train_dataset,val_dataset=random_split(full_dataset,[train_size,val_size])

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

if __name__ == "__main__":
    train_model()

