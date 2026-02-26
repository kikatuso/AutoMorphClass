import torch
import torch.nn as nn
import os
import glob
from time import time
from .models import Optic_Disc_Segmentation, Vessel_Segmentation,ArteryVeinSegmenter
from .feature_calculation import AutoMorphNumpyWrapper, Optic_Disc_Cup_Features,Vessel_Features
 
mask_prefixes = [
    "",                 # 1 general vessels
    "Vein_",            # 2
    "Artery_",          # 3
    "ZoneB_",           # 4
    "ZoneB_Vein_",      # 5
    "ZoneB_Artery_",    # 6
    "ZoneC_",           # 7
    "ZoneC_Vein_",      # 8
    "ZoneC_Artery_",    # 9
]


class AutoMorphModel(nn.Module):
    def __init__(self, return_as_tensor=True,lightweight=False):
        super().__init__()
        self.optic_disc_segmentator = Optic_Disc_Segmentation(resize=512,lightweight=lightweight)  
        self.vascular_segmentator = Vessel_Segmentation(resize=720,lightweight=lightweight) 
        self.artery_vein_segmentator = ArteryVeinSegmenter(resize=720,lightweight=lightweight)  
        self.optic_disc_cup_feature_calculator = AutoMorphNumpyWrapper(Optic_Disc_Cup_Features(),num_channels=2)
        self.vessel_feature_calculator = AutoMorphNumpyWrapper(Vessel_Features(),num_channels=1)
        self.return_as_tensor = return_as_tensor
        for param in self.parameters():
            param.requires_grad = False  
        self.eval()
    
    def get_height_width(self,mask):
        # mask shape: (B, H, W) or (H, W)
        index = torch.where(mask > 0)

        index_height = index[-2]
        index_width = index[-1]

        horizontal_width = index_width.max() - index_width.min()
        vertical_height = index_height.max() - index_height.min()

        return horizontal_width.item(), vertical_height.item()

    def _create_mask(self,mask, r1, r2, zone_centre):
        # binary mask shape: (B, H, W)
        B,_,H, W = mask.shape
        device = mask.device

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        cx, cy = zone_centre
        dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

        ring = (dist <= r2) & (dist > r1)
        ring = ring.float()

        # expand to match input mask shape
        ring = ring.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        return ring

    def define_zones(self,optic_disc_mask):
        # assuming optic_disc_mask shape: (B, C, H, W)
        disc = optic_disc_mask[:, 0:1, :, :]
        disc_horizontal_width, disc_vertical_height = self.get_height_width(disc)

        whole_index = torch.where(optic_disc_mask > 0)
        whole_index_height = whole_index[-2]
        whole_index_width = whole_index[-1]

        zone_centre = (
            whole_index_width.float().mean().long(),
            whole_index_height.float().mean().long()
        )

        radius = max(
            int(disc_horizontal_width / 2),
            int(disc_vertical_height / 2)
        )

        maskB = self._create_mask(disc, r1=2*radius, r2=3*radius, zone_centre=zone_centre)
        maskC = self._create_mask(disc, r1=3*radius, r2=5*radius, zone_centre=zone_centre)

        return maskB, maskC

    @torch.no_grad()
    def create_masks(self, x):
        optic_disc_mask = self.optic_disc_segmentator(x)[:,1:,:,:]  # take last two channels only; first channel is background
        zone_b, zone_c = self.define_zones(optic_disc_mask) # shape: (B, 1, H, W)
        vessel_mask = self.vascular_segmentator(x) # shape: (B, 1, H, W)
        artery_vein_mask = self.artery_vein_segmentator(x)[:,[0,2],:,:]  # take artery and vein channels only; red as vein, blue as artery
        all_vessels_mask = torch.cat([vessel_mask, artery_vein_mask], dim=1)
        orig = all_vessels_mask # shape: (B, 3, H, W)
        zone_b_part = all_vessels_mask * zone_b # shape: (B, 3, H, W)
        zone_c_part = all_vessels_mask * zone_c # shape: (B, 3, H, W)
        vessel_output = torch.cat([orig, zone_b_part, zone_c_part], dim=1) # shape: (B, 9, H, W)
        return vessel_output, optic_disc_mask
    
    def _to_tensor(self, features):
        keys = sorted(features.keys())
        feature_tensor = torch.stack([features[k] for k in keys], dim=1)
        return feature_tensor

    def forward(self, x):
        B = x.shape[0]
        vessel_output, optic_disc_mask = self.create_masks(x)
        optic_disc_cup_features = self.optic_disc_cup_feature_calculator(optic_disc_mask)

        vessel_long = vessel_output.reshape(-1, x.shape[2], x.shape[3]).unsqueeze(1) # shape: (B*9, 1, H, W)
        vessel_features_long = self.vessel_feature_calculator(vessel_long) # shape: (B*9, num_vessel_features)
        vessel_features = {}
        for key, value in vessel_features_long.items():
            value = value.view(B, 9)
            for i, prefix in enumerate(mask_prefixes):
                vessel_features[prefix + key] = value[:, i]

        features = {**optic_disc_cup_features, **vessel_features}
        if self.return_as_tensor:
            features = self._to_tensor(features)
        return features


# todo: add calibre features 

if __name__ == "__main__":

    print("Testing AutoMorphModel...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoMorphModel()
    model.to(device)

    from PIL import Image
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt

    img_path = '/gpfs3/well/papiez/users/zwk579/Results/DiffusionModels/stable_diffusion/logs/stable_diffusion/07_10_2025_cross_conditioning_light/samples/sample_0.png'

    img = Image.open(img_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).repeat(16, 1, 1, 1)  # shape: (3, 3, H, W)
    features = model(img_tensor.to(device))

    print(features.shape)


