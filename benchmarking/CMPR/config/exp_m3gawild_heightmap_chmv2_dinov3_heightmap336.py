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
    trained_image_model_name = 'Dinov3'
    image_embedding_dim = 384
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.1
    # image size (if int, resizes smaller side to size (preserves ratio). if tuple, resizes image directly)
    size = 336
    image_centre_crop = True
    range_mode = 'heightmap'
    range_image_size = (336, 336)  # note this will affect the spatial granularity of heightmap
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    dataloader = "M3GAWildDatasetRangeView"
    model = "CLIPModelV1NormM3GAWild"
    use_class_token = True
    
    # Cropping (max_range is max height for heightmap, not range)
    max_range = 65.536  # matching CHMv2 heightmap format (uint16 height in mm)
    heightmap_extent = (60, 60)  # 60x60m heightmap, matching aerial image with centrecrop
    gravity_align = True
    clip_invalid_depth = True

    # Dataset
    pos_radius: float = 15.0
    buffer_radius: float = 30.0
    neg_radius: float = 50.0
    only_ground_aerial: bool = True  # Only consider ground-aerial positive pairs, not ground-ground or aerial-aerial
    anchor_source: str = '_ground'  # Which viewpoint to use as queries ('_ground' or '_aerial')
    anchor_modality: str = 'lidar'  # Which modality to use as queries (images have more positives to sample)
    cloud_folder: str = 'clouds'  # Use full size pcls for denser depth images
    image_folder: str = 'chmv2_heightmaps_shrunk'