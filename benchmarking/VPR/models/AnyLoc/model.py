from typing import Optional

import torch 
import torch.nn as nn 

from models.lightning_model import LightningModel

from torchpack.utils.config import configs as CFG 

class DummyAggregator(nn.Module):
    '''
    Dummy aggregator so that we can use the LightningModel template, which asks for both 
    backbone and aggregator.
    '''
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x 

class AnyLocModel(LightningModel):
    def __init__(self):
        super().__init__()
        domain = CFG.domain
        if CFG.domain == 'map':
            # Will compute cluster centres on map prior to eval
            domain = None
        model = torch.hub.load('AnyLoc/DINO', 'get_vlad_model', domain=domain, backbone='DINOv2', device='cuda')
        self.backbone = model
        self.aggregator = DummyAggregator()

    # def cuda(self, device = None):
    #     # TorchHub version of AnyLoc is implemented poorly, need to manually move to GPU here
    #     self.backbone.cuda()
    #     self.backbone.vlad.c_centers.to(device)
    #     return super().cuda(device)

    # def to(self, *args, **kwargs):
    #     # TorchHub version of AnyLoc is implemented poorly, need to manually move to GPU here
    #     device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
    #     self.backbone.to(device)
    #     self.backbone.vlad.c_centers.to(device)
    #     return super().to(*args, **kwargs)

if __name__ == '__main__':
    # List of functions
    print(torch.hub.list('AnyLoc/DINO'))
    # Help about an individual function - like 'get_vlad_model'
    r = torch.hub.help('AnyLoc/DINO', 'get_vlad_model')
    print(r)

    # model = AnyLocModel()
    model = torch.hub.load('AnyLoc/DINO', 'get_vlad_model', domain='unstructured', backbone='DINOv2', device='cuda')
    img = torch.rand(16, 3, 224, 224).cuda()
    res = model(img)

    # TO FIT VLAD CLUSTER CENTRES:
    import einops as ein
    # Load model with no clusters (domain=None)
    model = torch.hub.load('AnyLoc/DINO', 'get_vlad_model', domain=None, backbone='DINOv2', device='cuda')
    # Extract features (typically the database images)
    res = model.extract(img)
    res_all = ein.rearrange(res, "B N D -> (B N) D")
    # Fit VLAD (to get cluster centers)
    model.fit(res_all)
    
    pass