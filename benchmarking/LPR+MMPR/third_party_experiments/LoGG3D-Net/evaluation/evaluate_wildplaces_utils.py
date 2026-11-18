
import os
import numpy as np
import tqdm
import torch
import pickle
from torchsparse import SparseTensor
from typing import Any, List
from torch.utils.data import DataLoader
import open3d as o3d
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
import math
import argparse
from torchpack.utils.config import configs 

QUERIES = {
    'cs_wildplaces': [
        'CSWildPlaces_Karawatha_evaluation_query.pickle',
        'CSWildPlaces_Venman_evaluation_query.pickle',
        'CSWildPlaces_QCAT_evaluation_query.pickle', 
        'CSWildPlaces_Samford_evaluation_query.pickle',
    ],
    'm3gawild': [
        'M3GAWild-aerial_Karawatha_evaluation_query.pickle',
        'M3GAWild-aerial_Venman_evaluation_query.pickle',
        'M3GAWild-aerial_QCAT_evaluation_query.pickle', 
        'M3GAWild-aerial_Samford_evaluation_query.pickle',
    ],
    'm3gawild_airborne': [
        'M3GAWild-airborne_Karawatha_evaluation_query.pickle',
        'M3GAWild-airborne_Venman_evaluation_query.pickle',
        'M3GAWild-airborne_QCAT_evaluation_query.pickle', 
        'M3GAWild-airborne_Samford_evaluation_query.pickle',
    ],
    'wildplaces': [
        'Karawatha_evaluation_query.pickle',
        'Venman_evaluation_query.pickle',
    ],
    'cs_campus3d': [
        'umd_evaluation_query.pickle',
    ],
    'mulran': [
        'DCC_10.0_5_query.pickle',
        'Sejong_0.2_20_query.pickle',
        'Riverside_10.0_5_query.pickle',
    ],
}
DATABASES = {
    'cs_wildplaces': [
        'CSWildPlaces_Karawatha_evaluation_database.pickle',
        'CSWildPlaces_Venman_evaluation_database.pickle',
        'CSWildPlaces_QCAT_evaluation_database.pickle', 
        'CSWildPlaces_Samford_evaluation_database.pickle',
    ],
    'm3gawild': [
        'M3GAWild-aerial_Karawatha_evaluation_database.pickle',
        'M3GAWild-aerial_Venman_evaluation_database.pickle',
        'M3GAWild-aerial_QCAT_evaluation_database.pickle', 
        'M3GAWild-aerial_Samford_evaluation_database.pickle',
    ],
    'm3gawild_airborne': [
        'M3GAWild-airborne_Karawatha_evaluation_database.pickle',
        'M3GAWild-airborne_Venman_evaluation_database.pickle',
        'M3GAWild-airborne_QCAT_evaluation_database.pickle', 
        'M3GAWild-airborne_Samford_evaluation_database.pickle',
    ],
    'wildplaces': [
        'Karawatha_evaluation_database.pickle',
        'Venman_evaluation_database.pickle',
    ],
    'cs_campus3d': [
        'umd_evaluation_database.pickle',
    ],
    'mulran': [
        'DCC_10.0_5_database.pickle',
        'Sejong_0.2_20_database.pickle',
        'Riverside_10.0_5_database.pickle',
    ],
}
LOCATION_NAMES = {
    'cs_wildplaces': [
        'Karawatha',
        'Venman',
        'QCAT', 
        'Samford',
    ],
    'm3gawild': [
        'Karawatha',
        'Venman',
        'QCAT', 
        'Samford',
    ],
    'm3gawild_airborne': [
        'Karawatha',
        'Venman',
        'QCAT', 
        'Samford',
    ],
    'wildplaces': [
        'Karawatha',
        'Venman',
    ],
    'cs_campus3d': [
        'umd',
    ],
    'mulran': [
        'DCC',
        'Sejong',
        'Riverside',
    ],
}

def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    if return_points:
        return inputs, feats
    else:
        return inputs

def sparcify_and_collate_list(list_data):

    if isinstance(list_data, SparseTensor):
        return list_data
    else:
        return sparse_collate(list_data)

class EvalDataset:
    def __init__(self, dataset, dataset_folder, cfg):
        self.set = dataset 
        self.dataset_folder = dataset_folder 
        self.cfg = cfg

    def __len__(self):
        return len(self.set)
    
    def get_pointcloud_tensor(self, fname):
        if 'wild' in self.cfg.eval_dataset_name:
            file_ext = os.path.splitext(fname)[-1]
            if file_ext == ".pcd":
                pc_o3d = o3d.io.read_point_cloud(fname)
                xyz = np.asarray(pc_o3d.points, dtype=np.float64)
                xyz = np.float32(xyz)
            elif file_ext == ".bin":
                xyz = np.fromfile(fname, dtype=np.float32)
                xyz = xyz.reshape(-1,4)
                # if not self.intensity:
                xyz = xyz[:,:3]
            else:
                raise NotImplementedError('Invalid point cloud type')
            # # downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            # xyz = np.asarray(pcd.points)
        elif self.cfg.eval_dataset_name == 'cs_campus3d':
            xyz = np.fromfile(fname, dtype=np.float64).reshape(-1, 3)
        else:
            raise NotImplementedError(f"No pointcloud loader for the dataset: '{self.cfg.eval_dataset_name}'")
        
        oo = np.ones(len(xyz)).reshape((-1,1))
        xyzr = np.hstack((xyz, oo)).astype(np.float32)

        return make_sparse_tensor(xyzr, self.cfg.voxel_size)

    def __getitem__(self, idx):
        # path =  os.path.join(self.dataset_folder, self.set[idx]['rel_scan_filepath'])
        path =  os.path.join(self.dataset_folder, self.set[idx]['query'])
        xyz0_th = self.get_pointcloud_tensor(path)

        return xyz0_th

def get_eval_dataloader(dataset, dataset_folder, cfg):
    dataset = EvalDataset(dataset, dataset_folder, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size = cfg.eval_batch_size,
        shuffle = False,
        collate_fn = sparcify_and_collate_list,#collate_sparse_tuple,
        num_workers = cfg.test_num_workers
    )
    return dataloader

def get_latent_vectors(model, dataset, dataset_folder, cfg, gd_dim=1024):
    if len(dataset) == 0:
        return None
    eval_dataloader = get_eval_dataloader(dataset, dataset_folder, cfg)
    vectors = []
    model.eval()
    
    for idx, batch in tqdm.tqdm(enumerate(eval_dataloader),
                                desc = 'Getting Latent Vectors',
                                total = math.ceil(len(eval_dataloader.dataset)
                                                  / cfg.eval_batch_size)):
        # batch = {k:v.to('cuda') for k,v in batch.items()}
        batch = batch.to('cuda')
        y = model(batch)
        gds = y[0].detach().cpu().numpy()
        gds = gds.reshape((-1,gd_dim))
        vectors.append(gds)

    vectors = np.concatenate(vectors, 0)
    return vectors 

# def get_latent_vectors_and_local(model, dataset, dataset_folder, cfg, gd_dim=1024):
#     if len(dataset) == 0:
#         return None
#     eval_dataloader = get_eval_dataloader(dataset, dataset_folder, cfg)
#     global_embeddings = None
#     local_embeddings = []
#     model.eval()
#     for idx, batch in tqdm.tqdm(enumerate(eval_dataloader),
#                                 desc = 'Getting Latent Vectors',
#                                 total = math.ceil(len(eval_dataloader.dataset)
#                                                   / cfg.eval_batch_size)):
#         # batch = {k:v.to('cuda') for k,v in batch.items()}
#         # TODO: Fix dataloader to output coords and points for use here
#         batch = batch.to('cuda')
#         global_embedding, keypoints, key_embeddings = compute_embedding(input, model, coords, points, is_dense=True)
#         if global_embeddings is None:
#             global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

#         global_embeddings[ndx] = global_embedding
#         local_embeddings.append({'keypoints': keypoints, 'features': key_embeddings})

#         return global_embeddings, local_embeddings

# def compute_embedding(pc, model, coords, points, is_dense=False):
#     """
#     Returns global embedding (np.array) as well as keypoints and corresponding descriptors (torch.tensors)
#     """
#     output_desc, output_feats, xc = model(pc)

#     if is_dense:
#         output_features = output_feats.cpu().detach().numpy()
#         output_points = points
#     else:
#         xc_coords = xc.C
#         xc_feat = xc.F.cpu().detach().numpy()
#         _, xc_counts = torch.unique(xc_coords[:, -1], return_counts=True)
#         y = torch.split(xc_coords, list(xc_counts))
#         assert len(y) == 1
#         xc_coords = xc_coords.cpu().detach().numpy()
#         xc_coords= xc_coords[:,:3]
#         len_coords = len(coords)
#         all_coords = np.vstack((coords,xc_coords))
#         all_hash_vals = ravel_hash(all_coords)

#         dense_hash = all_hash_vals[:len_coords]
#         coarse_hash = all_hash_vals[len_coords:]

#         sorter = np.argsort(dense_hash)
#         xc_hash_inds = sorter[np.searchsorted(dense_hash, coarse_hash, sorter=sorter)]

#         unique_xc_hash_inds, unique_xc_hash_inds_indices = np.unique(xc_hash_inds,
#                                             return_index=True)

#         output_points = points[unique_xc_hash_inds]
#         output_features = xc_feat[unique_xc_hash_inds_indices]

#     global_descriptor = output_desc.cpu().detach().numpy()
#     global_embedding = np.reshape(global_descriptor, (1, -1))

#     return global_embedding, torch.tensor(output_points, dtype=torch.float), torch.tensor(output_features, dtype=torch.float)

# def ravel_hash(x: np.ndarray) -> np.ndarray:
#     assert x.ndim == 2, x.shape

#     x -= np.min(x, axis=0)
#     x = x.astype(np.uint64, copy=False)
#     xmax = np.max(x, axis=0).astype(np.uint64) + 1

#     h = np.zeros(x.shape[0], dtype=np.uint64)
#     for k in range(x.shape[1] - 1):
#         h += x[:, k]
#         h *= xmax[k + 1]
#     h += x[:, -1]
#     return h

def load_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        file = pickle.load(f)
    return file 