import torch
import cv2
import sys
from src.model import SimpleCNN
import torch.nn.functional as F

model=SimpleCNN()
model.load_state_dict(torch.load("saved_models/mammogram_cnn.pth",map_location=torch.device('cpu')))
model.eval()

labels_map={0:"normal",1:"benign",2:"malignant"}
if len(sys.argv)!=2:
    print("Usage: python3 predict.py <image_path>")
    sys.exit(1)

image_path=sys.argv[1]
image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Failed to load the image from {image_path}")
    sys.exit(1)

image=cv2.resize(image,(224,224))

# Converts the image to a Tensor: 
# Adds a channel dimension (1) and a batch dimension (1), and normalizes to 0-1
image_tensor=torch.tensor(image,dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0

with torch.no_grad():
    output=model(image_tensor)
    probabilities=F.softmax(output,dim=1)
    predicted_label=torch.argmax(probabilities,dim=1).item()

print(f"Prediction: {labels_map[predicted_label]}")
print(f"probabilities: {probabilities.numpy()}")
