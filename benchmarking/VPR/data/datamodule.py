import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from data.dataset import WildCrossDataset, M3GAWildDataset
from torchpack.utils.config import configs as CFG 
from torchvision.transforms import v2  as T
import numpy as np 
import random 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class M3GAWildDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.m3gawild_path = CFG.m3gawild_path 
        
        self.pos_exclude_radius = CFG.pos_exclude_radius 
        self.pos_include_radius = CFG.pos_include_radius
        self.fov_include_radius = CFG.fov_include_radius 
        self.buffer_radius = CFG.buffer_radius
        self.only_ground_aerial = CFG.only_ground_aerial
        self.anchor_framerate_downsample = CFG.anchor_framerate_downsample 
        self.img_per_place = CFG.img_per_place 
        self.train_img_size = CFG.train_img_size 
        self.val_img_size = CFG.val_img_size 
        
        self.batch_size = CFG.batch_size 
        self.num_workers = CFG.num_workers 
        
        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.val_img_size, interpolation=3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def setup(self, stage=None):
        print(f'Setting up dataset: {self.m3gawild_path}')
        self.train_dataset = M3GAWildDataset(
            data_root=self.m3gawild_path,
            split='train',
            pos_exclude_radius=self.pos_exclude_radius,
            pos_include_radius=self.pos_include_radius,
            fov_include_radius=self.fov_include_radius,
            buffer_radius=self.buffer_radius,
            only_ground_aerial=self.only_ground_aerial,
            anchor_framerate_downsample=self.anchor_framerate_downsample,
            img_per_place=self.img_per_place,
            transform=self.train_transform,
        )
        self.val_dataset = M3GAWildDataset(
            data_root=self.m3gawild_path,
            split='val',
            pos_exclude_radius=self.pos_exclude_radius,
            pos_include_radius=self.pos_include_radius,
            fov_include_radius=self.fov_include_radius,
            buffer_radius=self.buffer_radius,
            only_ground_aerial=self.only_ground_aerial,
            anchor_framerate_downsample=self.anchor_framerate_downsample,
            img_per_place=self.img_per_place,
            transform=self.val_transform,
        )
        
    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(CFG.seed)
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = True,
            worker_init_fn = seed_worker,
            generator=g
        )

    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(CFG.seed)
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = False,
            worker_init_fn = seed_worker,
            generator=g
        )

class WildCrossDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.wildcross_path = CFG.wildcross_path 
        self.split_idx = CFG.split_idx 
        
        self.pos_exclude_radius = CFG.pos_exclude_radius 
        self.pos_include_radius = CFG.pos_include_radius
        self.fov_include_radius = CFG.fov_include_radius 
        self.anchor_framerate_downsample = CFG.anchor_framerate_downsample 
        self.img_per_place = CFG.img_per_place 
        self.train_img_size = CFG.train_img_size 
        
        self.batch_size = CFG.batch_size 
        self.num_workers = CFG.num_workers 
        
        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def setup(self, stage=None):
        print(self.wildcross_path)
        self.train_dataset = WildCrossDataset(
            data_root = self.wildcross_path,
            pos_exclude_radius = self.pos_exclude_radius,
            pos_include_radius = self.pos_include_radius,
            fov_include_radius = self.fov_include_radius,
            anchor_framerate_downsample = self.anchor_framerate_downsample,
            img_per_place = self.img_per_place,
            transform = self.train_transform,
            split_idx = self.split_idx 
        )
        
        self.val_datasets = []
        
    def train_dataloader(self):
        self.setup()
        g = torch.Generator()
        g.manual_seed(CFG.seed)
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = True,
            worker_init_fn = seed_worker,
            generator=g
        )

if __name__ == '__main__':
    import os
    # CFG.load(os.path.join(os.path.realpath(__file__), '../configs/BoQ.yaml'))
    # module = WildCrossDataModule()
    CFG.load(os.path.join(os.path.realpath(__file__), '../configs/m3gawild/BoQ.yaml'))
    module = M3GAWildDataModule()
    dl = module.train_dataloader()
