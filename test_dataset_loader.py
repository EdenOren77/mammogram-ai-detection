from src.dataset_loader import CustomDataset

dataset = CustomDataset(csv_path="data/image_data.csv")

print(f"Total samples: {len(dataset)}")
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")