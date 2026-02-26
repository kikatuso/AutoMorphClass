import os 
import sys
from pathlib import Path
import torch 
import torch.nn as nn
from torchvision.transforms.functional import resize
from torch.utils.checkpoint import checkpoint

from .conv_blocks import DoubleConv, Down, OutConv, Up_new

class SingleSegmenter(BaseSegmenter):
    def __init__(self,input_channels=3, n_filters = 32, n_classes=1, bilinear=False,pth_path=None,ignore_keys=[],use_checkpoint=True,
                 use_reentrant=False):
        super().__init__(input_channels, n_filters, n_classes, bilinear)
        self.use_reentrant = use_reentrant
        self.use_checkpoint = use_checkpoint
        if pth_path is not None:
            self.init_from_pth(pth_path, ignore_keys=ignore_keys)

    def init_from_pth(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu",weights_only=True)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False,)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    def _forward(self,x):
        mask_pred = super().forward(x)
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        return mask_pred_sigmoid
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=self.use_reentrant) if self.use_checkpoint else self._forward(x)

class Segmenter(nn.Module):
    def __init__(self,pth_path,input_channels=3, n_filters = 32, n_classes=1, bilinear=False,ignore_keys=[],lightweight=True,
                 return_soft_prob=False,use_checkpoint=True,use_reentrant=False,
                 resize_to_912=True):
        super().__init__()  
        self.use_reentrant = use_reentrant
        self.use_checkpoint = use_checkpoint
        self.return_soft_prob = return_soft_prob
        self.resize_to_912 = resize_to_912
        self.pth_files = os.listdir(pth_path)
        if lightweight: # only use the first segmenter
            self.pth_files = [self.pth_files[0]]
 
        self.models = nn.Sequential(*[
            SingleSegmenter(input_channels, n_filters, n_classes, bilinear, os.path.join(pth_path,pth_file,'G_best_F1_epoch.pth'), ignore_keys,use_checkpoint,use_reentrant) 
            for pth_file in self.pth_files])
            
    def preprocess_working(self, img, threshold=40.0):
        # Ensure the image tensor has the shape (N, 3, W, H)
        origW,origH = img.shape[-2:]
        W,H = max(origW,912),max(origH,912)
        img = resize(img, (W,H))
        if img.dim() == 3:
            img = img.unsqueeze(0)
        assert img.shape[1] == 3  # Check for 3 color channels
        
        # Convert threshold to 0-1 scale
        threshold = threshold / 255.0

        for i in range(img.shape[0]):
            # Mask for pixels where the first channel value is greater than threshold
            img_i = img[i]
            mask = img_i[0, :, :] > threshold
            img_i_masked = img_i[:, mask] 
            mean_i = img_i_masked.mean(dim=1)
            std_i = img_i_masked.std(dim=1)
            img_i = (img_i - mean_i[:, None, None]) / std_i[:, None, None]
            img[i] = img_i

        return img, origW, origH
    
    def preprocess(self, img, threshold=40.0):
        origW, origH = img.shape[-2:]
        if self.resize_to_912:
            W, H = max(origW, 912), max(origH, 912)
            img = resize(img, (W, H))
        else:
            W, H = origW, origH
        if img.dim() == 3:
            img = img.unsqueeze(0)
        assert img.shape[1] == 3  # Check for 3 color channels

        # Convert threshold to 0-1 scale
        threshold = threshold / 255.0

        # Create a mask for pixels where the first channel value is greater than the threshold
        mask = img[:, 0, :, :] > threshold  # Shape: (N, W, H)
        img_masked = img * mask.unsqueeze(1)

        # Calculate the mean and standard deviation of the masked pixels
        mean = img_masked.mean(dim=(2, 3),keepdim=True)
        std = img_masked.std(dim=(2, 3),keepdim=True)

        # Normalize the entire batch
        img = (img - mean) / std

        return img, origW, origH

    def _forward(self, x):
        x,origW,origH = self.preprocess(x)
        x_sum = sum(model(x) for model in self.models)/len(self.models)
        if self.resize_to_912:
            x_sum = resize(x_sum, (origW,origH))
        if not self.return_soft_prob:
            x_sum = (x_sum > 0.5).float()
        return x_sum
    
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=self.use_reentrant) if self.use_checkpoint else self._forward(x)



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