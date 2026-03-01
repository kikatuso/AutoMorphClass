import os
import glob
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from .modules.lwnet import get_arch


class Optic_Disc_Segmentation(nn.Module):
    def __init__(self,checkpoint_folder='checkpoints/optic_disc_and_cup/',verbose=False,mode='eval',lightweight=False,return_soft_prob=False,resize=512):
        super().__init__()
        self.mode = mode
        self.return_soft_prob = return_soft_prob
        self.resize = resize
        basedir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_folder = os.path.join(basedir, checkpoint_folder)
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '**', 'model_checkpoint.pth'), recursive=True)
        if lightweight:
            checkpoints = checkpoints[:1]  # Use only the first checkpoint for lightweight mode
        self.models = nn.ModuleList()  
        for i in range(len(checkpoints)):
            checkpoint = checkpoints[i] if checkpoint_folder is not None else None
            mi = Single_Segmentator(n_classes=3,model_checkpoint=checkpoint,mode=mode)
            self.models.append(mi)
        print(f"Optic_Disc_Segmentation initialised with {self._num_parameters():,} trainable parameters.") if verbose else ''
        if mode=='eval':
            for i in range(len(self.models)): 
                self.models[i].eval() # Set each segmentator to eval mode

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x): 
        if self.resize is not None:
            origW, origH = x.shape[-2:]
            x = resize(x, (self.resize, self.resize))
        if self.mode == 'eval':
            outputs = [m(x) for m in self.models]
            stacked = torch.stack(outputs, dim=0)  # Stack along new dimension
            pooled = torch.mean(stacked, dim=0)  # Average across models
        else:
            ridx = torch.randint(0, len(self.models), (1,)).item() # Randomly select a model index
            self.ridx = ridx
            pooled = self.models[ridx](x)
        if not self.return_soft_prob:
            pooled = (pooled > 0.5).float()
        if self.resize is not None:
            pooled = resize(pooled, (origW, origH))
        return pooled
        


class Single_Segmentator(nn.Module):
    def __init__(self,n_classes,in_c=3,model_checkpoint=None, model_name='wnet',verbose=False,mode='eval'):
        super().__init__() 
        self.model = get_arch(model_name,in_c=in_c,n_classes=n_classes,mode=mode)
        self.verbose = verbose
        self.mode = mode
        if model_checkpoint is not None:
            self.load_from_checkpoint(model_checkpoint)
    
    def load_from_checkpoint(self, checkpoint_path):
        print(f'Loading model from {checkpoint_path}...') if self.verbose else ''
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        else:
            weights = checkpoint
        self.model.load_state_dict(weights)  
    
    def get_last_layer(self):
        return self.model.get_last_layer()
    
    def forward(self, x):  # Added self
        return self.model(x)

if __name__ == "__main__":
    model = Optic_Disc_Segmentation(verbose=True,mode='eval',lightweight=False)
