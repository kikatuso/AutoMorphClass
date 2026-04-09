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
    def __init__(self, return_as_tensor=True,lightweight=False, savemask_path=None,include_zones=True):
        super().__init__()
        self.include_zones = include_zones
        self.optic_disc_segmentator = Optic_Disc_Segmentation(resize=512,lightweight=lightweight)  
        self.vascular_segmentator = Vessel_Segmentation(resize=912,lightweight=lightweight) 
        self.artery_vein_segmentator = ArteryVeinSegmenter(resize=720,lightweight=lightweight)  
        self.optic_disc_cup_feature_calculator = AutoMorphNumpyWrapper(Optic_Disc_Cup_Features(),num_channels=2,return_masks=True)
        self.vessel_feature_calculator = AutoMorphNumpyWrapper(Vessel_Features(),num_channels=1)
        self.return_as_tensor = return_as_tensor
        self.savemask_path = savemask_path
        for param in self.parameters():
            param.requires_grad = False  
        self.eval()
    
    def get_height_width(self, mask):
        # mask: (H, W) or (1, H, W)
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        idx = torch.where(mask > 0)
        if idx[0].numel() == 0:
            return 0, 0
        index_height = idx[0]
        index_width = idx[1]
        horizontal_width = (index_width.max() - index_width.min()).item()
        vertical_height = (index_height.max() - index_height.min()).item()
        return horizontal_width, vertical_height

    def _create_mask(self, H, W, r1, r2, zone_centre, device):
        """
        Create ring mask for a single image.
        Returns shape (1, 1, H, W)
        """
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )

        cx, cy = zone_centre

        dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

        ring = (dist <= r2) & (dist > r1)
        ring = ring.float()

        return ring.unsqueeze(0).unsqueeze(0)

    def define_zones(self, optic_disc_mask):
        B, C, H, W = optic_disc_mask.shape
        device = optic_disc_mask.device

        maskB_list = []
        maskC_list = []

        for b in range(B):
            disc = optic_disc_mask[b, 0]  # (H, W)

            disc_horizontal_width, disc_vertical_height = self.get_height_width(disc)

            if disc_horizontal_width == 0 or disc_vertical_height == 0:
                maskB_list.append(torch.zeros((1, 1, H, W), device=device))
                maskC_list.append(torch.zeros((1, 1, H, W), device=device))
                continue

            idx = torch.where(disc > 0)
            cy = int(idx[0].float().mean().item())
            cx = int(idx[1].float().mean().item())

            radius = max(disc_horizontal_width // 2, disc_vertical_height // 2)

            maskB = self._create_mask(H, W, 2*radius, 3*radius, (cx, cy), device)
            maskC = self._create_mask(H, W, 3*radius, 5*radius, (cx, cy), device)

            maskB_list.append(maskB)
            maskC_list.append(maskC)

        maskB = torch.cat(maskB_list, dim=0)
        maskC = torch.cat(maskC_list, dim=0)

        return maskB, maskC

    @torch.no_grad()
    def create_masks(self, x):
        optic_disc_mask = self.optic_disc_segmentator(x)[:,1:,:,:]  # take last two channels only; first channel is background
        vessel_mask = self.vascular_segmentator(x) # shape: (B, 1, H, W)
        artery_vein_mask = self.artery_vein_segmentator(x)[:,[0,2],:,:]  # take artery and vein channels only; red as vein, blue as artery
        all_vessels_mask = torch.cat([vessel_mask, artery_vein_mask], dim=1)
        if self.include_zones:
            zone_b, zone_c = self.define_zones(optic_disc_mask) # shape: (B, 1, H, W)
            orig = all_vessels_mask # shape: (B, 3, H, W)
            zone_b_part = all_vessels_mask * zone_b # shape: (B, 3, H, W)
            zone_c_part = all_vessels_mask * zone_c # shape: (B, 3, H, W)
            vessel_output = torch.cat([orig, zone_b_part, zone_c_part], dim=1) # shape: (B, 9, H, W)
        else:
            vessel_output = all_vessels_mask # shape: (B, 3, H, W)
        return vessel_output, optic_disc_mask
    
    def _to_tensor(self, features):
        keys = sorted(features.keys())
        feature_tensor = torch.stack([features[k] for k in keys], dim=1)
        return feature_tensor

    def plot_masks(self, savepath, masks, titles=["Vessels","Veins","Arteries","Zone B","Zone B Veins","Zone B Arteries","Zone C","Zone C Veins","Zone C Arteries"], img_names=None):
        os.makedirs(savepath, exist_ok=True)
        B, C, H, W = masks.shape
        assert C == len(titles), "Number of channels in masks should match number of titles"
        if img_names is not None:
            assert len(img_names) == B, "Number of image names should match batch size"
            names = [os.path.splitext(os.path.basename(n))[0] for n in img_names]
        else:
            names = [f"Sample{b}" for b in range(B)]
        # Create subfolders per title
        for title in titles:
            os.makedirs(os.path.join(savepath, title.replace(" ", "_")), exist_ok=True)
        for b in range(B):
            for c in range(C):
                folder = os.path.join(savepath, titles[c].replace(" ", "_"))
                fig = plt.figure(figsize=(5, 5))
                plt.imshow(masks[b, c].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(
                    os.path.join(folder, f"{names[b]}.png"),
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close(fig)
    
    def forward(self, x, x_names=None):
        B = x.shape[0]

        # --- Step 1: Create masks ---
        vessel_output, optic_disc_mask = self.create_masks(x)
        # --- Step 2: Optic disc features ---
        optic_disc_cup_features, optic_disc_mask = self.optic_disc_cup_feature_calculator(optic_disc_mask)
        # Stack disc features to shape (B, 6)
        disc_tensor = self._to_tensor(optic_disc_cup_features)

        # Per-image invalid mask (B,)
        invalid_disc_mask = (disc_tensor == -1).all(dim=1)

        # --- Step 3: Vessel features ---
        # vessel_output shape: (B, 9, H, W)
        vessel_long = vessel_output.reshape(-1, x.shape[2], x.shape[3]).unsqueeze(1)

        vessel_features_long = self.vessel_feature_calculator(vessel_long)
        # each value shape: (B*9,)

        vessel_features = {}

        for key, value in vessel_features_long.items():
            # reshape to (B, 9/3)
            nrows = 9 if self.include_zones else 3
            value = value.view(B, nrows)

            for i, prefix in enumerate(mask_prefixes):
                vessel_features[prefix + key] = value[:, i]

        # --- Step 4: Suppress ZoneB / ZoneC per invalid image ---
        for key in vessel_features:
            if key.startswith(("ZoneB_", "ZoneC_")):
                vessel_features[key][invalid_disc_mask] = -1

        # --- Step 5: Merge features ---
        features = {**optic_disc_cup_features, **vessel_features}
    
        if self.savemask_path is not None:
            self.plot_masks(self.savemask_path,vessel_output.clone(),img_names=x_names)
            self.plot_masks(self.savemask_path,optic_disc_mask.clone(),titles=["Optic Disc","Optic Cup"],img_names=x_names)

        # --- Step 6: Optional tensor conversion ---
        if self.return_as_tensor:
            features = self._to_tensor(features)
        return features

# todo: add calibre features 

