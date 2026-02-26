


class JointSegmentator(nn.Module):
    def __init__(self,checkpoint_folder='/gpfs3/well/papiez/users/zwk579/Results/RetinalLoss/model weights/main/MaskLoss_1_segmentator.pth'):
        super().__init__()

        self.optic_disc_segmentator = Optic_Disc_Segmentation()
        self.vascular_segmentator = Vessel_Segmentation()
        self.skeletonize = Skeletonisation()
        if checkpoint_folder is not None:
            self.load_from_checkpoint(checkpoint_folder)
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters
        self.eval()
        print(f"JointSegmentator initialised with {self._num_parameters():,} parameters.")

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def load_from_checkpoint(self, checkpoint_path):
        print(f'Loading JointSegmentator from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        else:
            weights = checkpoint
        self.load_state_dict(weights)

    def ste_binary(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        binary = (probs >= threshold).float()
        return binary + (probs - probs.detach())

    def forward(self, x):
        optic_disc_mask = self.optic_disc_segmentator(x)[:,1:,:,:]  # take last two channels only; first channel is background
        vessel_mask = self.vascular_segmentator(x)
        vessel_binary = self.ste_binary(vessel_mask)
        skeleton_mask = self.skeletonize(vessel_binary)

        return torch.cat([optic_disc_mask, vessel_mask, skeleton_mask], dim=1)  # Concatenate along channel dimension
