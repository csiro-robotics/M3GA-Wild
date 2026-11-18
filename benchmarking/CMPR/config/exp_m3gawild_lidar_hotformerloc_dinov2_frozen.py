import torch

class CFG:
    data_path = ""
    debug = False
    batch_size = 128
    num_workers = 16
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = 'Dinov2'
    trained_lidar_model_name = 'HOTFormerLoc'
    image_embedding_dim = 384
    lidar_embedding_dim = 256
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.1
    # image size (if int, resizes smaller side to size (preserves ratio). if tuple, resizes image directly)
    size = 336
    image_centre_crop = True
    range_mode = 'lidar'
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    dataloader = "M3GAWildDatasetRangeView"
    model = "CLIPModelV1NormM3GAWild"
    unfreeze_n_blocks = 2
    unfreeze_n_blocks_lidar = 2
    use_class_token = True
    
    # Cropping (max_range is max height for heightmap, not range)
    max_range = 50.0  # redundant for lidar
    gravity_align = True
    clip_invalid_depth = True

    # Dataset
    pos_radius: float = 15.0
    buffer_radius: float = 30.0
    neg_radius: float = 50.0
    only_ground_aerial: bool = True  # Only consider ground-aerial positive pairs, not ground-ground or aerial-aerial
    anchor_source: str = '_ground'  # Which viewpoint to use as queries ('_ground' or '_aerial')
    anchor_modality: str = 'lidar'  # Which modality to use as queries (images have more positives to sample)
    cloud_folder: str = 'clouds_voxel_0.80m_rmground'  # Use downsampled pcls for LiDAR encoder
    image_folder: str = 'images_shrunk'

    # HOTFormerLoc stuff
    coordinates: str = 'cylindrical'
    octree_depth: int = 7
    full_depth: int = 2
    normalize_points: bool = True
    aug_mode: int = 2