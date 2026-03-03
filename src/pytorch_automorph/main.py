import torch
import torch.nn as nn
import os
import glob
import matplotlib.pyplot as plt
from time import time
from .segmentation_models import Optic_Disc_Segmentation, Vessel_Segmentation,ArteryVeinSegmenter
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
    def __init__(self, return_as_tensor=True,lightweight=False, savemask_path=None):
        super().__init__()

        self.optic_disc_segmentator = Optic_Disc_Segmentation(resize=512,lightweight=lightweight)  
        self.vascular_segmentator = Vessel_Segmentation(resize=912,lightweight=lightweight) 
        self.artery_vein_segmentator = ArteryVeinSegmenter(resize=720,lightweight=lightweight)  
        self.optic_disc_cup_feature_calculator = AutoMorphNumpyWrapper(Optic_Disc_Cup_Features(),num_channels=2)
        self.vessel_feature_calculator = AutoMorphNumpyWrapper(Vessel_Features(),num_channels=1)
        self.return_as_tensor = return_as_tensor
        self.savemask_path = savemask_path
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
        if self.savemask_path is not None:
            self.plot_masks(self.savemask_path,vessel_output.clone())
            self.plot_masks(self.savemask_path,optic_disc_mask.clone(),titles=["Optic Disc","Optic Cup"])
        return vessel_output, optic_disc_mask
    
    def _to_tensor(self, features):
        keys = sorted(features.keys())
        feature_tensor = torch.stack([features[k] for k in keys], dim=1)
        return feature_tensor

    def plot_masks(self, savepath, masks, titles=["Vessels","Veins","Arteries","Zone B","Zone B Veins","Zone B Arteries","Zone C","Zone C Veins","Zone C Arteries"]):
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        B, C, H, W = masks.shape
        assert C == len(titles), "Number of channels in masks should match number of titles"

        for b in range(B):
            for c in range(C):
                fig = plt.figure(figsize=(5, 5))
                plt.imshow(masks[b, c].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(
                    os.path.join(savepath, f"Sample{b}_{titles[c].replace(' ','_')}.png"),
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close(fig)
    
    def forward(self, x):
        B = x.shape[0]

        # --- Step 1: Create masks ---
        vessel_output, optic_disc_mask = self.create_masks(x)

        # --- Step 2: Optic disc features ---
        optic_disc_cup_features = self.optic_disc_cup_feature_calculator(optic_disc_mask)

        # Stack disc features to shape (B, 6)
        disc_tensor = self._to_tensor(optic_disc_cup_features)

        # Per-image invalid mask (B,)
        invalid_disc_mask = (disc_tensor == -1).all(dim=1)

        # --- Step 3: Vessel features ---
        # vessel_output shape: (B, 9, H, W)
        vessel_long = vessel_output.reshape(-1, x.shape[2], x.shape[3]).unsqueeze(1)
        # shape: (B*9, 1, H, W)

        vessel_features_long = self.vessel_feature_calculator(vessel_long)
        # each value shape: (B*9,)

        vessel_features = {}

        for key, value in vessel_features_long.items():
            # reshape to (B, 9)
            value = value.view(B, 9)

            for i, prefix in enumerate(mask_prefixes):
                vessel_features[prefix + key] = value[:, i]

        # --- Step 4: Suppress ZoneB / ZoneC per invalid image ---
        for key in vessel_features:
            if key.startswith(("ZoneB_", "ZoneC_")):
                vessel_features[key][invalid_disc_mask] = -1

        # --- Step 5: Merge features ---
        features = {**optic_disc_cup_features, **vessel_features}

        # --- Step 6: Optional tensor conversion ---
        if self.return_as_tensor:
            features = self._to_tensor(features)

        return features

# todo: add calibre features 

