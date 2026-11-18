# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------
# Adapted from BoQ by Ethan Griffiths (Data61, Pullenvale)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.minkloc_multimodal import GeM

class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
        use_cls_token=False,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        self.use_cls_token = use_cls_token
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # unfreeze the last few blocks
        for block in self.dino.blocks[ -unfreeze_n_blocks : ]:
            for param in block.parameters():
                param.requires_grad = True
        
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
            x = blk(x)

        if self.use_cls_token:
            return x[:, 0]
        else:
            x = x[:, 1:] # remove the [CLS] token

        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape # or C = self.embed_dim
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x
    
    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Freeze required layers
            for layer in self.net[:nb_layers - unfreeze_n_blocks]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0

        # Output channels
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels

    def forward(self, x):
        return self.net(x)

class ImageFeatExtractorWrapper(nn.Module):
    """Wrapper for MinkLoc++ image feature extractors."""
    def __init__(
        self,
        out_channels: int,
        backbone_name='dinov2_vits14',
        unfreeze_n_blocks: int = 2,
        pool_method='GeM',
        normalize_embeddings: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.pool_method = pool_method
        self.normalize_embeddings = normalize_embeddings

        if 'resnet' in self.backbone_name:
            backbone = ResNet(
                backbone_name=self.backbone_name,
                unfreeze_n_blocks=self.unfreeze_n_blocks,
                crop_last_block=True,
            )
        elif 'dinov2' in self.backbone_name:
            print('Overriding DINOv2 pooling method: using CLS token instead')
            self.pool_method = 'CLS'
            use_cls_token = self.pool_method.lower() == 'cls'
            backbone = DinoV2(
                backbone_name=self.backbone_name,
                unfreeze_n_blocks=self.unfreeze_n_blocks,
                use_cls_token=use_cls_token,
            )
        self.backbone = backbone
        self.projection = nn.Linear(backbone.out_channels, out_channels)

        if self.pool_method.lower() == 'cls':
            self.pool = nn.Identity()
        elif self.pool_method.lower() == 'gem':
            self.pool = GeM()
        else:
            raise ValueError(f'Invalid pooling method: {self.pool_method}')

    def forward(self, x: dict):
        x = self.backbone(x['images'])
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)
        return {'global': x}