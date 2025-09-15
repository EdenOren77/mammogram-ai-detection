import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    def __init__(self,csv_path,transform=None):
        self.data=pd.read_csv(csv_path)
        self.transform=transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row=self.data.iloc[index]
        image_path=row["path"]
        label=row["label"]
        image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) #An array (NumPy ndarray) of size [H, W] of image pixels

        #Resize image to a fixed size (224x224)to avoid stack errors
        image=cv2.resize(image,(224,224))
        
        # .unsqueeze(0)- Adding channel 1 for Grayscale
        # We will perform Normalization- Dividing by 255 so that the network learns better 
        image=torch.tensor(image,dtype=torch.float32).unsqueeze(0) /255.0
        label=torch.tensor(label,dtype=torch.long)
        if self.transform:
            image=self.transform(image)
        return image,label