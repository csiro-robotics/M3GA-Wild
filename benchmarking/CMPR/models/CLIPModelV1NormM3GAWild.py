from pathlib import Path
import yaml
import timm
import torch
import torch.nn.functional as F
from torch import nn
from easydict import EasyDict as edict
from .HOTFormerLoc.model_factory import model_factory
from utils.torch_utils import to_device

HOTFORMERLOC_CFG_PATH = Path(__file__).parent / 'HOTFormerLoc/hotformerloc_wild-places_cfg.yaml'
HOTFORMERLOC_WEIGHTS_PATH = Path(__file__).parent / '../weights/hotformerloc_wild-places.pth'
HOTFORMERLOC_WEIGHTS_URL = 'https://www.dropbox.com/scl/fi/yd94iy9dq6k1m312ifnyx/hotformerloc_wild-places.pth?rlkey=5ndv0p48c7hyjvah90eab1l1e&st=zl1716hh&dl=0'

class DinoWrapperV2(nn.Module):
    def __init__(self, dino, unfreeze_n_blocks=2):
        super().__init__()
        self.dino = dino 
        self.out_channels = self.dino.embed_dim
        if unfreeze_n_blocks is None:
            unfreeze_n_blocks = 2
        assert unfreeze_n_blocks > 0
        self.unfreeze_n_blocks = unfreeze_n_blocks
        
        for p in self.dino.parameters():
            p.requires_grad = False 
            
        for block in self.dino.blocks[-self.unfreeze_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        
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

        # Get class token
        x = x[:,0]
        return x 


class DinoWrapperV3(nn.Module):
    def __init__(self, dino, unfreeze_n_blocks=None):
        super().__init__()
        self.dino = dino 
        self.unfreeze_n_blocks = unfreeze_n_blocks 
        
        if self.unfreeze_n_blocks is not None and self.unfreeze_n_blocks > 0:
            for p in self.dino.parameters():
                p.requires_grad = False 
                
            for block in self.dino.blocks[-self.unfreeze_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Unfreeze final norm layer
            for p in self.dino.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.dino(x)


class ResNet50Wrapper(nn.Module):
    def __init__(self, model, unfreeze_n_layers=0):
        super().__init__()
        self.model = model 
        self.unfreeze_n_layers = unfreeze_n_layers
        self.layers = ['layer1', 'layer2', 'layer3', 'layer4']
        
        if self.unfreeze_n_layers is not None and self.unfreeze_n_layers > 0:
            for p in self.model.parameters():
                p.requires_grad = False 
                
            for layer in self.layers[-self.unfreeze_n_layers:]:
                for p in getattr(self.model, layer).parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable, unfreeze_n_blocks=None,
    ):
        super().__init__()
        if model_name == 'Dinov2':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model = DinoWrapperV2(dino, unfreeze_n_blocks=unfreeze_n_blocks)
            
        elif model_name == 'Dinov3':
            # Here define REPO_DIR and CHECKPOINT_PATH to your local dinov3 repo and checkpoint paths
            # See https://github.com/facebookresearch/dinov3 for more details
            REPO_DIR="/path/to/dinov3"
            CHECKPOINT_PATH="/path/to/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            
            dino = torch.hub.load(REPO_DIR,
                                  'dinov3_vits16',
                                  source='local',
                                  weights=CHECKPOINT_PATH)
        
            self.model = DinoWrapperV3(dino, unfreeze_n_blocks=unfreeze_n_blocks)

        elif model_name == 'Dinov3_sat493m':
            # Here define REPO_DIR and CHECKPOINT_PATH to your local dinov3 repo and checkpoint paths
            # See https://github.com/facebookresearch/dinov3 for more details
            REPO_DIR="/path/to/dinov3"
            CHECKPOINT_PATH="/path/to/dinov3/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            
            dino = torch.hub.load(REPO_DIR,
                                  'dinov3_vitl16',
                                  source='local',
                                  weights=CHECKPOINT_PATH)
        
            self.model = DinoWrapperV3(dino, unfreeze_n_blocks=unfreeze_n_blocks)
        
        
        else:
            self.model = timm.create_model(
                model_name, pretrained, num_classes=0, global_pool="avg"
            )
            # for p in self.model.parameters():
            #     p.requires_grad = trainable
            if 'resnet' not in model_name.lower():
                raise ValueError()
            self.model = ResNet50Wrapper(self.model, unfreeze_n_layers=unfreeze_n_blocks)

    def forward(self, x):
        return self.model(x)


class LidarEncoder(nn.Module):
    """
    Encode lidar point clouds to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, unfreeze_n_blocks=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        if model_name == 'HOTFormerLoc':
            with open(HOTFORMERLOC_CFG_PATH, 'r') as f:
                hotformerloc_cfg = edict(yaml.safe_load(f))
            self.model = model_factory(hotformerloc_cfg)
            if self.pretrained:
                if not HOTFORMERLOC_WEIGHTS_PATH.is_file():
                    raise FileNotFoundError(
                        f"No pre-trained weights found at '{HOTFORMERLOC_WEIGHTS_PATH}'. "
                        f"Download them first and place them in the 'weights/' folder (link: {HOTFORMERLOC_WEIGHTS_URL})."
                    )
                self.model.load_state_dict(torch.load(HOTFORMERLOC_WEIGHTS_PATH))

            if self.unfreeze_n_blocks is not None and self.unfreeze_n_blocks > 0:
                for p in self.model.parameters():
                    p.requires_grad = False 
                    
                hotf_stage = self.model.backbone.backbone.hotf_stage
                for block in hotf_stage.rtsa_blocks[-self.unfreeze_n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for hosa_level in hotf_stage.hosa_blocks:
                    for block in hosa_level[-self.unfreeze_n_blocks:]:
                        for p in block.parameters():
                            p.requires_grad = True

                # Unfreeze pooling layer
                for p in self.model.pooling.parameters():
                    p.requires_grad = True
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.device=CFG.device
        self.temperature=CFG.temperature
        self.unfreeze_n_blocks = getattr(CFG, 'unfreeze_n_blocks', None)
        self.unfreeze_n_blocks_lidar = getattr(CFG, 'unfreeze_n_blocks_lidar', None)
        self.range_mode = getattr(CFG, 'range_mode', 'range')
        if self.range_mode not in ('range', 'heightmap', 'lidar'):
            raise ValueError('Invalid range image mode')

        self.encoder_camera = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable, unfreeze_n_blocks=self.unfreeze_n_blocks)
        self.projection_camera = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        if self.range_mode == 'lidar':
            self.encoder_lidar = LidarEncoder(model_name=CFG.trained_lidar_model_name, pretrained=CFG.pretrained, unfreeze_n_blocks=self.unfreeze_n_blocks_lidar)
            self.projection_lidar = ProjectionHead(embedding_dim=CFG.lidar_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        else:
            self.encoder_lidar = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable, unfreeze_n_blocks=self.unfreeze_n_blocks)
            self.projection_lidar = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        self.ignore_index = -100
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.neg_thresh = 50.0

    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        camera_image_features = self.encoder_camera(batch["camera_image"])
        if self.range_mode == 'lidar':
            lidar_features = self.encoder_lidar(batch)['global']
        else:
            lidar_features = self.encoder_lidar(batch["lidar_image"])
                
        # Getting camera Image and lidar range Embeddings (with same dimension)
        camera_image_embeddings = self.projection_camera(camera_image_features)
        lidar_embeddings = self.projection_lidar(lidar_features)

        # Normalize the embeddings to have unit norm
        camera_image_embeddings = F.normalize(camera_image_embeddings, dim=-1)
        lidar_embeddings = F.normalize(lidar_embeddings, dim=-1)

        # Calculating the Loss
        # Getting ignore matrix 
        cross_dist = torch.cdist(batch['lidar_coord'], batch['camera_coord'])
        negative_mask = cross_dist < self.neg_thresh
        negative_mask.fill_diagonal_(False)
        
        logits = (lidar_embeddings @ camera_image_embeddings.T) / self.temperature
        logits[negative_mask] = -10000.0 # Set non-negatives to a low number so they're ignored when calculating softmax 
        
        # Compute CLIP loss, with added padding for sampled aerial negatives if needed
        targets = torch.arange(logits.size(0), device='cuda')
        targets[logits.size(1):] = self.ignore_index
        lidar_loss = self.cross_entropy(logits, targets)
        targets = torch.arange(logits.size(1), device='cuda')
        targets[logits.size(0):] = self.ignore_index
        camera_loss = self.cross_entropy(logits.T, targets)
        
        loss_total = (lidar_loss + camera_loss) / 2.0 
        
        # Calculate number past filter 
        with torch.no_grad():
            lidar_softmax_logits = F.softmax(logits, dim = -1)
            camera_softmax_logits = F.softmax(logits.T, dim = -1)
            
            average_diagonal = (torch.diagonal(lidar_softmax_logits) + torch.diagonal(camera_softmax_logits)) / 2.0
            num_past_filter = torch.sum(average_diagonal >= 0.95) / len(average_diagonal)
        
        return loss_total, num_past_filter

    def get_camera_embeddings(self, batch):
        image_features = self.encoder_camera(batch["camera_image"].to(self.device))
        image_embeddings = self.projection_camera(image_features)

        # Normalize the embeddings to have unit norm
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        return image_embeddings

    def get_lidar_embeddings(self, batch):
        if self.range_mode == 'lidar':
            lidar_features = self.encoder_lidar(to_device(batch, device=self.device, construct_octree_neigh=True))['global']
        else:
            lidar_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        lidar_embeddings = self.projection_lidar(lidar_features)

        # Normalize the embeddings to have unit norm
        lidar_embeddings = F.normalize(lidar_embeddings, dim=-1)

        return lidar_embeddings
