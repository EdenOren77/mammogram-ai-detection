import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import SimpleCNN
from src.dataset_loader import CustomDataset
import os
from torch.utils.data import random_split

#Training configuration
batch_size=16
learning_rate=0.001
num_epochs=10

# Device configuration (a graphics card if available)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

#Load the full dataset
csv_path="data/image_data.csv"
full_dataset=CustomDataset(csv_path)

#split the data- 80% for train and 20% for the validation
train_size=int(0.8*len(full_dataset))
val_size=len(full_dataset)-train_size
train_dataset,val_dataset=random_split(full_dataset,[train_size,val_size])

#dataLoader
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True) #shuffle=True so that the network does not learn a fixed order
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

#init the model
model=SimpleCNN().to(device)

#define loss function and optimizer
criterion=nn.CrossEntropyLoss() #normal/benign/malignant
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#training loop
for epoch in range(num_epochs):
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
        total +=labels.size(0)
        correct+=(predicted==labels).sum().item()
    
    train_accuracy=100*correct/total
    avg_loss=running_loss/len(train_loader)

    #validation
    model.eval()
    val_correct=0
    val_total=0

    with torch.no_grad():
        for images,labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            val_total+=labels.size(0)
            val_correct+=(predicted==labels).sum().item()
            
    val_accuracy=100*val_correct/val_total
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Val Acc: {val_accuracy:.2f}%")
#save the trained model
os.makedirs("saved_models",exist_ok=True)
torch.save(model.state_dict(),"saved_models/mammogram_cnn.pth")


