import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

curr_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)
from model import BreastCancerModel

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]
CLASS_NAMES=['Normal','Benign','Malignant']

MODEL_PATH=os.path.join(os.path.dirname(curr_dir), "saved_models","resnet50_mammogram.pth")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Predictor:
    def __init__(self):
        self.model=BreastCancerModel()
        try:
            # Load weights safely, mapping to the correct device (CPU/GPU)
            checkpoint=torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(DEVICE)
            # disables Dropout/BatchNorm learning
            self.model.eval()
        except Exception as e:
            raise e
        
        self.transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])

    def predict(self,image_path):
        try:
            #Load Image and Convert to RGB (Crucial for handling grayscale inputs)
            image=Image.open(image_path).convert('RGB')
            image_tensor=self.transform(image)
            image_tensor=image_tensor.unsqueeze(0)
            # Move data to the same device as the model
            image_tensor=image_tensor.to(DEVICE)

            with torch.no_grad():
                outputs=self.model(image_tensor)
                probs=F.softmax(outputs,dim=1)
                confidence,predicted_class=torch.max(probs,1)
                label=CLASS_NAMES[predicted_class.item()]
                score=confidence.item()*100
            
            return{
                "label":label,
                "confidence":round(score,2),
                "all_probabilities":{
                        name: round(prob.item()*100,2) 
                        for name,prob in zip(CLASS_NAMES,probs[0])
                    }
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

predictor=Predictor()