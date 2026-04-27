import os 
import sys
import glob
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from .modules.conv_blocks import DoubleConv, Down, OutConv, Up_new


class Vessel_Segmentation(nn.Module):
    def __init__(self,checkpoint_folder='checkpoints/vessel_segmentation/',
                input_channels=3, n_filters = 32, n_classes=1, bilinear=False,ignore_keys=[],lightweight=False,return_soft_prob=False,resize=720,remove_border=True):
        super().__init__()  
        self.return_soft_prob = return_soft_prob
        self.resize = resize
        self.remove_border = remove_border
        basedir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_folder = os.path.join(basedir, checkpoint_folder)
        self.pth_files = glob.glob(os.path.join(checkpoint_folder, '**', 'G_best_F1_epoch.pth'), recursive=True)
        if lightweight: # only use the first segmenter
            print('Vessel_Segmentation: lightweight mode enabled, using only the first checkpoint.')
            self.pth_files = [self.pth_files[0]]
 
        self.models = nn.Sequential(*[
            SingleSegmenter(input_channels, n_filters, n_classes, bilinear, os.path.join(pth_path,pth_file), ignore_keys) 
            for pth_path, pth_file in zip([os.path.dirname(pth_file) for pth_file in self.pth_files], self.pth_files)])
            

    def preprocess(self, img, threshold=40.0/255.0):
        # Blue channel zero-check: if blue channel is all zero, duplicate green channel
        if img[:, 2, :, :].sum() == 0:
            img = img[:, 1:2, :, :].expand(-1, 3, -1, -1).clone()

        # Threshold is on raw pixel scale (matching Code 1, no /255 conversion)
        mask = img[:, 0, :, :] > threshold  # Shape: (N, W, H)

        # Compute mean/std only over masked (foreground) pixels, per channel
        mask_expanded = mask.unsqueeze(1).expand_as(img)  # (N, C, W, H)
        
        mean = img[mask_expanded].view(img.shape[1], -1).mean(dim=1).view(1, img.shape[1], 1, 1)
        std  = img[mask_expanded].view(img.shape[1], -1).std(dim=1).view(1, img.shape[1], 1, 1)

        # Normalize entire image using foreground statistics
        img = (img - mean) / std
        return img
    
    def _resize(self, img):
        origW, origH = img.shape[-2:]
        if self.resize is not None:
            W, H = max(origW, self.resize), max(origH, self.resize)
            img = resize(img, (W, H))
        else:
            W, H = origW, origH
        if img.dim() == 3:
            img = img.unsqueeze(0)
        assert img.shape[1] == 3  # Check for 3 color channels
        return img, origW, origH


    def _remove_border(self, img, r = 20):
        img_gray = img.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # threshold
        mask = (img_gray > 10/255.0).float()
        k = 2 * r + 1
        eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)
        return eroded
        
        

    def forward(self, x):
        x, origW, origH = self._resize(x)
        x_processed = self.preprocess(x)
        mask_preds = [model(x_processed) for model in self.models]
        x_sum = sum(mask_preds) / len(mask_preds)
        if not self.return_soft_prob:
            x_sum = (x_sum > 0.5).float()
        if self.remove_border:
            eroded_mask = self._remove_border(x)
            x_sum = x_sum * eroded_mask
        return x_sum
    

def disk_kernel(radius):
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij"
    )
    mask = (x**2 + y**2) <= radius**2
    return mask.float()

class BaseSegmenter(nn.Module):
    def __init__(self, input_channels, n_filters, n_classes, bilinear=False):
        super(BaseSegmenter, self).__init__()

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
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class SingleSegmenter(BaseSegmenter):
    def __init__(self,input_channels=3, n_filters = 32, n_classes=1, bilinear=False,pth_path=None,ignore_keys=[],verbose=False):
        super().__init__(input_channels, n_filters, n_classes, bilinear)
        if pth_path is not None:
            self.init_from_pth(pth_path, ignore_keys=ignore_keys,verbose=verbose)

    def init_from_pth(self, path, ignore_keys=list(),verbose=False):
        sd = torch.load(path, map_location="cpu",weights_only=True)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))  if verbose else ''
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False,)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys") if verbose else None
        if len(missing) > 0:
            print(f"Missing Keys: {missing}") if verbose else None
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}") if verbose else None
    
    def forward(self,x):
        mask_pred = super().forward(x)
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        return mask_pred_sigmoid

if __name__=='__main__':
    model = Vessel_Segmentation()
    