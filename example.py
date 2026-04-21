from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from src.pytorch_automorph import AutoMorphModel
from time import time
from tqdm import tqdm


class RepeatImageDataset(Dataset):
    def __init__(self, img_path, repeat=32):
        self.img = Image.open(img_path).convert('RGB')
        self.tensor = ToTensor()(self.img)
        self.repeat = repeat

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        return self.tensor


def run_example(model, device, loader):
    for batch in tqdm(loader, desc="Processing batches"):
        batch = batch.to(device)
        features = model(batch)


def time_example(repeats=10, batch_size=16):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = AutoMorphModel(return_as_tensor=False, lightweight=True)
    model = model.to(device)

    dataset = RepeatImageDataset('example_images/image3.png', repeat=batch_size*repeats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Using device: {device}")

    start = time()
    run_example(model, device, loader)
    end = time()
    t_total = (end - start)

    print(f"Average time per batch: {t_total:.4f} seconds")
    model.vessel_feature_calculator.print_timing_summary()

if __name__ == "__main__":


    time_example(repeats=10, batch_size=16)
