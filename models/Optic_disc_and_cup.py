import os
import glob
import torch
import torch.nn as nn

from .modules.lwnet import get_arch


class Optic_Disc_Segmentation(nn.Module):
    def __init__(self,verbose=False,mode='eval',checkpoint_folder=None,num_segmentators=1):
        super().__init__()
        self.mode = mode
        if checkpoint_folder is None:
            checkpoints = [None]*num_segmentators
        else:
            checkpoints = glob.glob(os.path.join(checkpoint_folder, '**', 'model_checkpoint.pth'), recursive=True)
            assert len(checkpoints)>=num_segmentators, f"Found only {len(checkpoints)} checkpoints, but num_segmentators={num_segmentators}."
        self.models = nn.ModuleList()  
        for i in range(num_segmentators):
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
        if self.mode == 'eval':
            outputs = [m(x) for m in self.models]
            stacked = torch.stack(outputs, dim=0)  # Stack along new dimension
            return torch.mean(stacked, dim=0)  # Average across models
        else:
            ridx = torch.randint(0, len(self.models), (1,)).item() # Randomly select a model index
            self.ridx = ridx
            return self.models[ridx](x)



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
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        else:
            weights = checkpoint
        self.model.load_state_dict(weights)  
    
    def get_last_layer(self):
        return self.model.get_last_layer()
    
    def forward(self, x):  # Added self
        return self.model(x)
