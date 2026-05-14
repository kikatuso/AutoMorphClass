import torch
import torch.nn as nn
import os
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from torchvision.transforms.functional import resize as TF_resize
from torchvision.transforms import InterpolationMode
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
    def __init__(self, return_as_tensor=True,lightweight=False, savemask_path=None,include_zones=True,include_artery_vein=True,include_optic_disc=True):
        super().__init__()
        self.include_zones = include_zones
        self.include_artery_vein = include_artery_vein
        self.include_optic_disc = include_optic_disc
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
    def create_masks(self, x, vessel_mask=None, optic_disc_mask=None, artery_vein_mask=None):
        # --- Vessel (always required) ---
        if vessel_mask is None:
            vessel_mask = self.vascular_segmentator(x)
        else:
            vessel_mask = self._resize_to_920(vessel_mask)

        parts = [vessel_mask]

        # --- Artery / Vein ---
        if self.include_artery_vein:
            if artery_vein_mask is None:
                artery_vein_mask = self.artery_vein_segmentator(x)[:, [0, 2]]
            artery_vein_mask = self._resize_to_920(artery_vein_mask)
            parts.append(artery_vein_mask)
        all_vessels = torch.cat(parts, dim=1)

        # --- Optic disc ---
        if self.include_optic_disc or self.include_zones:
            if optic_disc_mask is None:
                optic_disc_mask = self.optic_disc_segmentator(x)[:, 1:]
            optic_disc_mask = self._resize_to_920(optic_disc_mask)
        else:
            optic_disc_mask = None

        # --- Zones ---
        if self.include_zones:
            if optic_disc_mask is None:
                raise ValueError("Zones require optic disc mask")

            zone_b, zone_c = self.define_zones(optic_disc_mask)
            if not self.include_optic_disc:
                optic_disc_mask = None 
            base = all_vessels
            all_vessels = torch.cat([
                base,
                base * zone_b,
                base * zone_c
            ], dim=1)

        return all_vessels, optic_disc_mask
    
    def _to_tensor(self, features):
        keys = sorted(features.keys())
        feature_tensor = torch.stack([features[k] for k in keys], dim=1)
        return feature_tensor

    def plot_masks(self, savepath, masks, titles=["Vessels","Veins","Arteries","Zone B","Zone B Veins","Zone B Arteries","Zone C","Zone C Veins","Zone C Arteries"], img_names=None):
        
        def fetch_last_samplename(savepath):
            files = glob.glob(os.path.join(savepath, "*", "*.png"))
            nums = []

            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                m = re.search(r"Sample(\d+)", name)
                if m:
                    nums.append(int(m.group(1)))

            return max(nums) + 1 if nums else 0

        os.makedirs(savepath, exist_ok=True)
        B, C, H, W = masks.shape
        assert C == len(titles), "Number of channels in masks should match number of titles"
        if img_names is not None:
            assert len(img_names) == B, "Number of image names should match batch size"
            names = [os.path.splitext(os.path.basename(n))[0] for n in img_names]
        else:
            b_start = fetch_last_samplename(savepath)
            names = [f"Sample{b}" for b in range(b_start, b_start + B)]
        # Create subfolders per title
        for title in titles:
            os.makedirs(os.path.join(savepath, title.replace(" ", "_")), exist_ok=True)
        for b in range(B):
            for c in range(C):
                folder = os.path.join(savepath, titles[c].replace(" ", "_"))
                img = masks[b, c].detach().cpu().numpy()
                img = (img * 255).astype('uint8')
                Image.fromarray(img).save(os.path.join(folder, f"{names[b]}.png"))

    def _resize_to_920(self, mask):
        return TF_resize(mask, (912, 912), interpolation=InterpolationMode.NEAREST)
    
    @torch.no_grad()
    def forward(
        self,
        x,
        x_names=None,
        vessel_masks=None,
        optic_disc_masks=None,
        artery_vein_masks=None
    ):
        B = x.shape[0]

        # --- Step 1: Masks (with overrides) ---
        vessel_output, optic_disc_mask = self.create_masks(
            x,
            vessel_mask=vessel_masks,
            optic_disc_mask=optic_disc_masks,
            artery_vein_mask=artery_vein_masks
        )

        features = {}

        # --- Step 2: Optic disc features ---
        if self.include_optic_disc:
            disc_feats, optic_disc_mask = self.optic_disc_cup_feature_calculator(optic_disc_mask)
            features.update(disc_feats)

            disc_tensor = self._to_tensor(disc_feats)
            invalid_disc_mask = (disc_tensor == -1).all(dim=1)
        else:
            invalid_disc_mask = torch.zeros(B, dtype=torch.bool, device=x.device)

        # --- Step 3: Vessel features ---
        B, C, H, W = vessel_output.shape
        vessel_long = vessel_output.reshape(B * C, H, W).unsqueeze(1)

        vessel_features_long = self.vessel_feature_calculator(vessel_long)

        # dynamic prefixes
        prefixes = ["Vessel"]
        if self.include_artery_vein:
            prefixes += ["Artery", "Vein"]

        if self.include_zones:
            base = prefixes.copy()
            prefixes = base + [f"ZoneB_{p}" for p in base] + [f"ZoneC_{p}" for p in base]

        for key, value in vessel_features_long.items():
            value = value.view(B, C)
            for i, prefix in enumerate(prefixes):
                features[f"{prefix}_{key}"] = value[:, i]

        # --- Step 4: Invalidate zones ---
        if self.include_zones:
            for k in features:
                if k.startswith(("ZoneB_", "ZoneC_")):
                    features[k][invalid_disc_mask] = -1

        # --- Step 5: Optional plotting ---
        if self.savemask_path is not None:

            titles = ["Vessels"]

            if self.include_artery_vein:
                titles += ["Veins", "Arteries"]

            if self.include_zones:
                titles += ["Zone B"]
                if self.include_artery_vein:
                    titles += ["Zone B Veins", "Zone B Arteries"]

                titles += ["Zone C"]
                if self.include_artery_vein:
                    titles += ["Zone C Veins", "Zone C Arteries"]

            self.plot_masks(self.savemask_path, vessel_output.clone(), img_names=x_names, titles=titles)

            if self.include_optic_disc:
                self.plot_masks(
                    self.savemask_path,
                    optic_disc_mask.clone(),
                    titles=["Optic Disc", "Optic Cup"],
                    img_names=x_names
                )

        # --- Step 6: Optional tensor ---
        if self.return_as_tensor:
            features = self._to_tensor(features)

        return features

# todo: add calibre features 

