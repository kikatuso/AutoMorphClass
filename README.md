# AutoMorphClass

A PyTorch wrapper for automated retinal image analysis, built on top of the [AutoMorph](https://github.com/rmaphoh/AutoMorph) pipeline.

## Overview

`AutoMorphClass` provides a clean, plug-and-play PyTorch interface for extracting quantitative vascular features from retinal fundus images. It bundles the full AutoMorph segmentation pipeline into a single callable `nn.Module`, making it easy to integrate into existing research or clinical workflows.

The pipeline performs:

- **Optic disc segmentation**
- **Vessel segmentation**
- **Artery/vein segmentation**
- **Quantitative retinal vascular feature extraction**

---

## Installation

### Option 1 — Install as a package (recommended)
```bash
git clone https://github.com/kikatuso/AutoMorphClass.git
cd AutoMorphClass
pip install -e .
```

Then import directly in your code:
```python
from pytorch_automorph import AutoMorphModel
```

### Option 2 — Use in place

Clone the repo and install dependencies manually:
```bash
git clone https://github.com/kikatuso/AutoMorphClass.git
cd AutoMorphClass
pip install torch torchvision pillow scikit-image opencv-python
```

Then import relative to the repo root:
```python
from src import AutoMorphModel
```

> For CUDA-specific PyTorch builds, see [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install .`.

---

## Quick Start
```python
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from pytorch_automorph import AutoMorphModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
img = Image.open("example_images/image2.png").convert("RGB")
x = ToTensor()(img).unsqueeze(0).to(device)

# Initialize model
model = AutoMorphModel(return_as_tensor=True).to(device)

# Extract features
features = model(x)
print(features.shape)  # (B, F)
```

To get named features as a dictionary:
```python
model = AutoMorphModel(return_as_tensor=False).to(device)
features_dict = model(x)

for name, value in features_dict.items():
    print(name, value)
```

---

## API Reference

### `AutoMorphModel(return_as_tensor=True, lightweight=False)`

A PyTorch `nn.Module` that wraps the full AutoMorph retinal analysis pipeline.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_as_tensor` | `bool` | `True` | If `True`, returns a feature tensor of shape `(B, F)`. If `False`, returns a `dict` mapping feature names to scalar tensors. |
| `lightweight` | `bool` | `False` | Uses smaller segmentation backbones when `True`, trading accuracy for speed. |

**Input**

- A `torch.Tensor` of shape `(B, 3, H, W)` — a batch of RGB retinal images.
- The tensor must reside on the same device as the model.

**Output**

- `return_as_tensor=True` → `torch.Tensor` of shape `(B, F)`
- `return_as_tensor=False` → `dict[str, torch.Tensor]` of scalar feature values

> The model runs in **evaluation mode** and is designed for inference only.

---

## Dependencies

- `torch`
- `torchvision`
- `pillow`
- `scikit-image`
- `opencv-python`

---

## Example Notebook

An end-to-end usage example is provided in [`example.ipynb`](example.ipynb), demonstrating how to load images, run the model, and interpret the extracted vascular features.

---

## Reference

This project is based on the original AutoMorph framework:

> **AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline**  
> https://github.com/rmaphoh/AutoMorph

---

## License

This project is licensed under the [MIT License](LICENSE).