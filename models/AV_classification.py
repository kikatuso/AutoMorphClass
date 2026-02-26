import os 
import sys
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import re

from .modules.conv_blocks import DoubleConv, Down, Up, OutConv, Up_new, side_one, side_two, side_three


class ArteryVeinSegmenter(nn.Module):
    def __init__(self,
                seed_path='/well/papiez/users/zwk579/Analysis/AutoMorphClass/checkpoints/AV_classification/',
                input_channels=3, n_filters = 32, n_classes=4, bilinear=False,lightweight=False,eval=True,resize=720,verbose=False):
        super().__init__()  
        self.resize = resize # this is following the original implementation 
        self.pth_files = self.find_folders(seed_path)
        self.n_classes = n_classes
        if lightweight: # only use the first segmenter
            self.pth_files = [self.pth_files[0]]
 
        self.models = nn.Sequential(*[BaseSegmenter(input_channels, n_filters, n_classes, bilinear, pth_path=str(pth_file),verbose=verbose) for pth_file in self.pth_files])
        if eval:
            print(f"Setting {len(self.models)} segmenter(s) to eval mode.") if verbose else ''
            self.eval()
        
    def find_folders(self, base):
        base = Path(base)
        folders = sorted(p / "Discriminator_unet" for p in base.iterdir() if p.is_dir()
            and re.search(r'randomseed_\d+', p.name)
            and (p / "Discriminator_unet").is_dir())
        return folders

    def _fix_scale(self, x):
        # x: (B, C, H, W)
        if x.max() <= 1.0:
            x = x * 255.0
        # foreground mask based on red channel
        mask = x[:, 0:1, :, :] > 0
        # avoid empty mask
        if mask.any():
            mean = (x * mask).sum(dim=(2,3), keepdim=True) / mask.sum(dim=(2,3), keepdim=True).clamp(min=1)
            std  = torch.sqrt(
                ((x - mean)**2 * mask).sum(dim=(2,3), keepdim=True)
                / mask.sum(dim=(2,3), keepdim=True).clamp(min=1))
        else:
            mean = x.mean(dim=(2,3), keepdim=True)
            std  = x.std(dim=(2,3), keepdim=True)
        x = (x - mean) * std   # replicate their bug intentionally
        return x

    def forward(self, x):
        x = self._fix_scale(x)
        if self.resize is not None:
            oH, oW = x.shape[2], x.shape[3]
            x = TF.resize(x, [self.resize, self.resize], interpolation=TF.InterpolationMode.BILINEAR)
        mean = sum(model(x) for model in self.models) / len(self.models)
        pred = mean.argmax(dim=1, keepdim=True)  # (B,1,H,W)
        artery  = (pred == 1).float()
        vein    = (pred == 2).float()
        overlap = (pred == 3).float()
        output = torch.cat([artery, overlap, vein], dim=1) * 255.0
        if self.resize is not None:
            output = TF.resize(output, [oH, oW], interpolation=TF.InterpolationMode.NEAREST)
        return output



class BaseSegmenter(nn.Module):
    def __init__(self, input_channels=3, n_filters=32, n_classes=4, bilinear=False,pth_path=None,verbose=True):
        super().__init__()
        self.G = Generator_main(input_channels, n_filters, n_classes, bilinear)
        self.G_A = Generator_branch(input_channels, n_filters, n_classes, bilinear)
        self.G_V = Generator_branch(input_channels, n_filters, n_classes, bilinear)
        if pth_path is not None:
            self.init_from_pth(pth_path,verbose=verbose)

    def init_from_pth(self, path_folder,
                    path_dict = {'G':'CP_best_F1_all.pth','G_A':'CP_best_F1_A.pth','G_V':'CP_best_F1_V.pth'}, 
                    ignore_prefixes = (),verbose=False
                    ):
        for model_name, pth_file in path_dict.items():
            model = getattr(self, model_name)
            path = os.path.join(path_folder, pth_file)
            sd = torch.load(path, map_location="cpu",weights_only=True)
            keys = list(sd.keys())
            for k in keys:
                sd = {k: v for k, v in sd.items() if not k.startswith(ignore_prefixes)}
            missing, unexpected = model.load_state_dict(sd, strict=False,)
            missing = [k for k in missing if not k.startswith(ignore_prefixes)]
            print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys") if verbose else None
            if len(missing) > 0:
                print(f"Missing Keys: {missing}") if verbose else None
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}") if verbose else None
    
    def forward(self, x):
        _, masks_pred_G_fusion_A = self.G_A(x)
        _, masks_pred_G_fusion_V = self.G_V(x)
        mask_pred, _, _, _ = self.G(x,masks_pred_G_fusion_A,masks_pred_G_fusion_V)
        mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
        return mask_pred_softmax
    

class Generator_main(nn.Module):
    def __init__(self, input_channels, n_filters, n_classes, bilinear=False):
        super(Generator_main, self).__init__()

        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, n_filters)
        self.down1 = Down(n_filters, 2*n_filters)
        self.down2 = Down(2*n_filters, 4*n_filters)
        self.down3 = Down(4*n_filters, 8*n_filters)
        self.down4 = Down(8*n_filters, 16*n_filters)

        self.downsample = nn.MaxPool2d(2)

        self.up1 = Up_new(16*n_filters, 8*n_filters, bilinear)
        self.S1 = side_one(8*n_filters, n_classes)

        self.up2 = Up_new(8*n_filters, 4*n_filters, bilinear)
        self.S2 = side_two(4*n_filters, n_classes)

        self.up3 = Up_new(4*n_filters, 2*n_filters, bilinear)
        self.S3 = side_three(2*n_filters, n_classes)

        self.up4 = Up_new(2*n_filters, 1*n_filters, bilinear)
        
        #self.outc = OutConv((n_filters+8), n_classes)
        self.outc = OutConv((n_filters), n_classes)


    def forward(self, x, x_a, x_v):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        s1 = self.S1(x)
        x = self.up2(x, x3)
        s2 = self.S2(x)
        
        #x = torch.cat([x_a, x, x_v], dim=1)
        x = self.up3(x, x2)
        s3 = self.S3(x)
        x = self.up4(x, x1)
        
        x_fusion = torch.mean(torch.stack([x_a, x, x_v],dim=0),dim=0)
        logits = self.outc(x_fusion)
        #logits = self.outc(x)

        return logits, s1, s2, s3


class Generator_branch(nn.Module):
    def __init__(self, input_channels, n_filters, n_classes, bilinear=False):
        super(Generator_branch, self).__init__()

        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, n_filters)
        self.down1 = Down(n_filters, 2*n_filters)
        self.down2 = Down(2*n_filters, 4*n_filters)
        self.down3 = Down(4*n_filters, 8*n_filters)
        self.down4 = Down(8*n_filters, 16*n_filters)

        self.up1 = Up_new(16*n_filters, 8*n_filters, bilinear)
        self.up2 = Up_new(8*n_filters, 4*n_filters, bilinear)
        self.up3 = Up_new(4*n_filters, 2*n_filters, bilinear)
        self.up4 = Up_new(2*n_filters, 1*n_filters, bilinear)
        self.outc = OutConv(n_filters, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_final = self.up4(x, x1)
        logits = self.outc(x_final)
        return logits,x_final



if __name__ == "__main__":
    # Example usage


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ArteryVeinSegmenter()
    model.to(device)

    from torchvision.transforms import ToTensor
    from torchvision.transforms import Resize
    to_tensor = ToTensor()


    path = '/gpfs3/well/papiez/users/zwk579/Results/DiffusionModels/stable_diffusion/logs/stable_diffusion/07_10_2025_cross_conditioning_light/samples'
    img_path = f"{path}/AUTOMORPH/M1/Good_quality/sample_0.png" # AUTOMORPH/M2/binary_vessel/raw/
    from PIL import Image
    import numpy as np

    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256))  # match training size

    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        seg = model(img_tensor)
