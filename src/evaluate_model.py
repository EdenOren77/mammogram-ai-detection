import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import sys
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from model import BreastCancerModel
from dataset_loader import CustomDataset

BATCH_SIZE = 16
DATA_PATH = os.path.join(os.path.dirname(curr_dir), "data", "image_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(curr_dir), "saved_models", "resnet50_mammogram.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def evaluate_and_plot():
    print(f"Loading model from {MODEL_PATH}...")
    
    model = BreastCancerModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    dataset = CustomDataset(DATA_PATH, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    print("Running predictions...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Normal', 'Benign', 'Malignant']

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Breast Cancer Detection')

    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_and_plot()