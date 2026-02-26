import torch
import torch.nn as nn
import os
import glob

from .modules.lwnet import get_arch
from .Optic_disc_and_cup import Single_Segmentator

class Skeletonisation(Single_Segmentator):
    def __init__(self,model_checkpoint='/well/papiez/users/zwk579/Analysis/AutoMorphClass/checkpoints/skeletonisation/skel_checkpoint.pth',verbose=False,mode='eval'):
        super().__init__(n_classes=1,in_c=1,model_checkpoint=model_checkpoint,mode=mode)
        print(f"Skeletonisation initialised with {self._num_parameters():,} trainable parameters.") if verbose else ''
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # Added self
        return super().forward(x)


if __name__ == "__main__":

    model = Skeletonisation()
