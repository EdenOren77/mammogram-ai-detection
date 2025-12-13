import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image 

class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row=self.data.iloc[index]
        image_path=row["path"]
        label=row["label"]

        if not os.path.exists(image_path):
            fixed_path = os.path.join("..",image_path)
            if os.path.exists(fixed_path):
                image_path = fixed_path
            else:
                raise FileNotFoundError(f"Image not found at: {image_path} nor at {fixed_path}")

        image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        if image is None:
             raise ValueError(f"Failed to load image via cv2: {image_path}")

        image_pil=Image.fromarray(image).convert("RGB")
        if self.transform:
            image_tensor=self.transform(image_pil)
        else:
            import torchvision.transforms as T
            base_transform=T.Compose([
                T.Resize((224,224)),
                T.ToTensor()
            ])
            image_tensor=base_transform(image_pil)
        label_tensor=torch.tensor(label,dtype=torch.long)
            
        return image_tensor,label_tensor