import torchvision.transforms as T
import PIL.Image as Image 
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch 
import einops as ein
from tqdm import tqdm 

class EvalDataset:
    def __init__(self, filenames, image_size, centre_crop=False):
        self.filenames = filenames 
        transforms = [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if centre_crop:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            if not image_size[0] == image_size[1]:
                raise ValueError('Expected square image size for centre crop')
            transforms.append(T.Resize(size=image_size[0], antialias=True))  # Resize short edge
            transforms.append(T.CenterCrop(size=image_size))
        else:
            transforms.append(T.Resize(size=image_size, antialias=True))
        self.transform = T.Compose(transforms)
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        im = Image.open(self.filenames[idx]).convert("RGB")
        im = self.transform(im)
        return im 

@torch.no_grad()
def compute_cluster_centres(env_data, image_size, model: torch.nn.Module, debug=False, batch_size=32, vlad_subsample_rate=2, centre_crop=False):
    """
    Compute AnyLoc VLAD cluster centres on the database.
    """
    all_feats = []
    for seq_data in env_data:
        seq_dataset = EvalDataset(filenames=seq_data['filenames'][::vlad_subsample_rate], image_size=image_size, centre_crop=centre_crop)
        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    pin_memory=True,
                                    shuffle=False)
        if debug:
            raise NotImplementedError()
        else:
            for batch in tqdm(seq_dataloader, desc=f"Extracting features for Seq. {seq_data['name']}"):
                batch = batch.to('cuda')
                feats = model.backbone.extract(batch)
                all_feats.append(feats)
    all_feats = torch.cat(all_feats)
    all_feats = ein.rearrange(all_feats, "B N D -> (B N) D")
    print('Computing VLAD cluster centers')
    model.backbone.fit(all_feats)
    del all_feats

@torch.no_grad()
def get_descriptors_positions(env_data, image_size, model: torch.nn.Module, debug=False, batch_size=32, centre_crop=False):
    info = []
    for seq_data in env_data:
        seq_dataset = EvalDataset(filenames=seq_data['filenames'], image_size=image_size, centre_crop=centre_crop)
        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    pin_memory=True,
                                    shuffle=False)
        seq_feats = []
        if debug:
            seq_feats = F.normalize(torch.randn(len(seq_dataset), 12288), dim = -1)
        else:
            for batch in tqdm(seq_dataloader, desc=f"Extracting features for Seq. {seq_data['name']}"):
                batch = batch.to('cuda')
                feats = model.forward(batch)
                seq_feats.append(feats.cpu())
            seq_feats = torch.cat(seq_feats)
        seq_pos = torch.tensor(seq_data['coords'])
        seq_ts = torch.tensor(seq_data['timestamps'])
        info.append({'name': seq_data['name'], 'feats': seq_feats, 'coords': seq_pos, 'timestamps': seq_ts, 'filenames': seq_data['filenames']})
    return info 
