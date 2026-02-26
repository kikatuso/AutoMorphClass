# AutoMorphClass

A PyTorch wrapper for automated retinal image analysis based on the AutoMorph pipeline.

`AutoMorphClass` performs:

- Optic disc segmentation  
- Vessel segmentation  
- Artery/vein segmentation  
- Extraction of quantitative retinal vascular features  

Designed for simple integration into PyTorch workflows and research pipelines.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kikatuso/AutoMorphClass.git
cd AutoMorphClass
```

Install dependencies:

```bash
pip install torch torchvision pillow
```

Make sure you install the correct PyTorch version for your CPU or CUDA setup.

---

## Usage

```python
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from src import AutoMorphModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
img = Image.open("example_images/image2.png").convert("RGB")
x = ToTensor()(img).unsqueeze(0).to(device)

# Initialize model
model = AutoMorphModel(return_as_tensor=True).to(device)

# Extract features
features = model(x)
print(features.shape)
```

To return features as a dictionary instead of a tensor:

```python
model = AutoMorphModel(return_as_tensor=False).to(device)
features_dict = model(x)

for name, value in features_dict.items():
    print(name, value)
```

---

## API

### `AutoMorphModel(return_as_tensor=True, lightweight=False)`

**Parameters**

- `return_as_tensor`  
  - `True`: returns features as a tensor of shape `(B, F)`  
  - `False`: returns a dictionary `{feature_name: value}`  

- `lightweight`  
  - Uses smaller segmentation backbones if available  

**Input**

- Tensor of shape `(B, 3, H, W)`  
- Must be on the same device as the model  

**Output**

- Feature tensor `(B, F)`  
  or  
- Dictionary of scalar tensors  

The model runs in evaluation mode and is intended for inference.

---

## Project Structure

```text
AutoMorphClass/
│
├── src/
│   ├── __init__.py
│   └── main.py
│
├── example.py
├── example_images/
└── pyproject.toml
```

---

## Reference

Based on the original AutoMorph framework:

https://github.com/rmaphoh/AutoMorph

---

## License

MIT License