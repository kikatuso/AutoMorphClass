from PIL import Image
import os 
import torch
from torchvision.transforms import ToTensor
from src import AutoMorphModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = 'example_images/image3.png'

img = Image.open(img_path).convert('RGB')
img_tensor = ToTensor()(img).unsqueeze(0).to(device)
model = AutoMorphModel(return_as_tensor=True)
model.to(device)
features = model(img_tensor)
print(features.shape)

## if you wish to return dictionary 
model = AutoMorphModel(return_as_tensor=False)
model.to(device)
features = model(img_tensor)
for key, value in features.items():
    print(f"{key}: {value.item()}")