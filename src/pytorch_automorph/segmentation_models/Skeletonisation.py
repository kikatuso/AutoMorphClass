import torch
import torch.nn as nn
import os
import glob

from .modules.lwnet import get_arch
from .Optic_disc_and_cup import Single_Segmentator

class Skeletonisation(Single_Segmentator):
    def __init__(self,model_checkpoint='checkpoints/skeletonisation/skel_checkpoint.pth',verbose=False,mode='eval',return_soft_prob=False):
        basedir = os.path.dirname(os.path.abspath(__file__))
        model_checkpoint = os.path.join(basedir, model_checkpoint)
        super().__init__(n_classes=1,in_c=1,model_checkpoint=model_checkpoint,mode=mode)
        self.return_soft_prob = return_soft_prob
        print(f"Skeletonisation initialised with {self._num_parameters():,} trainable parameters.") if verbose else ''
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # Added self
        out = super().forward(x)
        if not self.return_soft_prob:
            return (out > 0.5).float()
        return out


if __name__ == "__main__":

    model = Skeletonisation()
