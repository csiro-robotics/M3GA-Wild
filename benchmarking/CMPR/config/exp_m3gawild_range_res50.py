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
    trained_image_model_name = 'resnet50'
    image_embedding_dim = 2048
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.1
    # image size (if int, resizes smaller side to size (preserves ratio). if tuple, resizes image directly)
    size = 288
    image_centre_crop = True
    range_image_size = (224, 896)  # orig (210, 900)
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    dataloader = "M3GAWildDatasetRangeView"
    model = "CLIPModelV1NormM3GAWild"
    use_class_token = False
    
    # Cropping
    crop = False
    crop_distance = False
    distance_threshold = 30.0
    max_range = 50.0
    gravity_align = True
    clip_invalid_depth = False

    # Dataset
    pos_radius: float = 15.0
    buffer_radius: float = 30.0
    neg_radius: float = 50.0
    only_ground_aerial: bool = True  # Only consider ground-aerial positive pairs, not ground-ground or aerial-aerial
    anchor_source: str = '_ground'  # Which viewpoint to use as queries ('_ground' or '_aerial')
    anchor_modality: str = 'lidar'  # Which modality to use as queries (images have more positives to sample)
    cloud_folder: str = 'clouds'  # Use full size pcls for denser depth images
    image_folder: str = 'images_shrunk'