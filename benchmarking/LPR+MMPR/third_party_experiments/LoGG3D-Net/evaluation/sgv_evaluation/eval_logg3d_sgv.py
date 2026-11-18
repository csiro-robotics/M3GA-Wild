# This script is adapted from: https://github.com/csiro-robotics/SpectralGV/blob/main/evaluation/LoGG3D-Net/eval_logg3d_sgv.py

import argparse
import numpy as np
import tqdm
import os
import sys
import random
from typing import List
import open3d as o3d
import time
import copy
import torch
from torchsparse import SparseTensor
from itertools import repeat
from typing import List, Tuple, Union
from torchsparse.utils.collate import sparse_collate
import pickle

from pose_utils import relative_pose, mulran_relative_pose, apply_transform
from point_cloud_utils import icp, two_stage_icp, make_open3d_feature, make_open3d_point_cloud, preprocess_pointcloud, get_pointcloud_loader
from sgv_utils import sgv_fn

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.eval_config import get_config_eval
from models.pipeline_factory import get_pipeline
from evaluation.evaluate_wildplaces_utils import load_from_pickle, QUERIES, DATABASES, LOCATION_NAMES

print('\n' + ' '.join([sys.executable] + sys.argv))

DEBUG = False
DEBUG_MAX_QUERIES = 10
DEBUG_CSWP_FAILURE = False

class Evaluator:
    # def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
    def __init__(self, dataset_root: str, dataset_type: str, device: str, 
                 radius: List[float] = (5, 20), k: int = 50, n_samples: int =None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        # self.eval_set_filepath = os.path.join(os.path.dirname(__file__), '../../datasets/',self.dataset_type, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug

        # TODO: Convert to current (v3) query/db format, and ensure it supports Wild-Places + other changes made in evaluate_wildplaces.py
        # self.eval_set = EvaluationSet()
        # self.eval_set.load(self.eval_set_filepath)
        # if debug:
        #     # Make the same map set and query set in debug mdoe
        #     self.eval_set.map_set = self.eval_set.map_set[:4]
        #     self.eval_set.query_set = self.eval_set.map_set[:4]

        # if n_samples is None or len(self.eval_set.query_set) <= n_samples:
        #     self.n_samples = len(self.eval_set.query_set)
        # else:
        self.n_samples = n_samples

        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: dict, model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e['query'])
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings

def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def average_nested_dict(nested_dict: dict):
    """
    Returns the average values for all keys in a nested dictionary.  
    E.g. {'sample1': {'metric1': {'submetric1': val1}}, 'sample2': {'metric1': {'submetric1': val2}}, ...}
    returns {'metric1': {'submetric1': np.mean([val1, val2, ...]), ...}}
    """
    assert isinstance(nested_dict, dict)
    average_dict = {}
    if len(nested_dict) == 0:
        return average_dict
    sub_dicts = list(nested_dict.values())
    assert all([isinstance(sub_dict_ii, dict) for sub_dict_ii in sub_dicts])
    # Remove any metrics with 'failure' in the key (e.g. rr_failures, failure_query...)
    metrics = [metric for metric in sub_dicts[0].keys() if 'failure' not in str(metric)]
    sub_metrics = [sub_metric for sub_metric in sub_dicts[0][metrics[0]].keys() if 'failure' not in str(sub_metric)]
    for metric_ii in metrics:
        average_dict[metric_ii] = {}
        temp_metric_list = {key: [] for key in sub_metrics}
        for sub_dict_jj in sub_dicts:
            assert metric_ii in sub_dict_jj, 'Invalid nested dict, child keys must match'
            for sub_metric_kk in sub_metrics:
                temp_metric_list[sub_metric_kk].append(sub_dict_jj[metric_ii][sub_metric_kk])
        for sub_metric_kk in sub_metrics:
            average_dict[metric_ii][sub_metric_kk] = np.mean(temp_metric_list[sub_metric_kk], axis=0)
    return average_dict

class MetLocEvaluator(Evaluator):
    # def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
    def __init__(self, dataset_root: str, dataset_type: str, device: str, 
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5, voxel_size: float = 0.1,
                 icp_refine: bool = True, local_max_eval_threshold: float = np.inf, debug: bool = False,
                 num_points: int = 150000, sequence_idx: List[int] = None):
        # super().__init__(dataset_root, dataset_type, eval_set_pickle, device, radius, k, n_samples, debug=debug)
        super().__init__(dataset_root, dataset_type, device, radius, k, n_samples, debug=debug)
        self.repeat_dist_th = repeat_dist_th
        self.icp_refine = icp_refine
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.local_max_eval_threshold = local_max_eval_threshold
        self.sequence_idx = sequence_idx

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, d_thresh, *args, **kwargs):
        skip_same_run = ('wild' in self.dataset_type) 
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False

        # Run evaluation on a single dataset
        global_metrics, local_metrics = {}, {}

        queries = QUERIES[cfg.eval_dataset_name]
        databases = DATABASES[cfg.eval_dataset_name]
        location_names = LOCATION_NAMES[cfg.eval_dataset_name]
        assert len(queries) == len(databases), f'Length of Query and Database Dictionaries is not the same: {len(queries)} vs {len(databases)}'
        
        for ii, (query_sets_path, database_sets_path, location) in enumerate(zip(queries, databases, location_names)):
            if self.sequence_idx is not None and ii not in self.sequence_idx:
                continue
            # save_path = os.path.dirname(__file__) + '/pickles/logg3d_' + self.dataset_type + '/'
            save_path = os.path.join(os.path.dirname(__file__), 'pickles', f'logg3d_{self.dataset_type}', location)
            os.makedirs(save_path, exist_ok=True)

            # Load run information, feature vectors 
            query_sets = load_from_pickle(os.path.join(self.dataset_root, query_sets_path))
            map_sets = load_from_pickle(os.path.join(self.dataset_root, database_sets_path))

            query_embeddings, query_local_embeddings = [], []
            map_embeddings, map_local_embeddings, map_positions = [], [], []

            ######################## DEBUGGING #################################
            if DEBUG:
                query_sets = [{k: v for k, v in query_set.items() if k < DEBUG_MAX_QUERIES} for query_set in query_sets]
            ######################## DEBUGGING #################################

            # COMMENT OUT BLOCK WHEN LOADING FROM PICKLE
            for query_set in query_sets:
                temp_query_embeddings, temp_query_local_embeddings = self.compute_embeddings(query_set, model) 
                query_embeddings.append(temp_query_embeddings)
                query_local_embeddings.append(temp_query_local_embeddings)
            # with open(os.path.join(save_path, 'query_embeddings.pickle'), 'wb') as handle:
            #     pickle.dump(query_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(os.path.join(save_path, 'query_local_embeddings.pickle'), 'wb') as handle:
            #     pickle.dump(query_local_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # # UNCOMMENT BLOCK WHEN LOADING FROM PICKLE
            # with open(os.path.join(save_path, 'query_embeddings.pickle'), 'rb') as handle:
            #     query_embeddings = pickle.load(handle)
            # with open(os.path.join(save_path, 'query_local_embeddings.pickle'), 'rb') as handle:
            #     query_local_embeddings = pickle.load(handle)
                
            # COMMENT OUT BLOCK WHEN LOADING FROM PICKLE
            for map_set in map_sets:
                temp_map_embeddings, temp_map_local_embeddings = self.compute_embeddings(map_set, model) 
                temp_map_positions = np.array([(x['northing'], x['easting']) for x in map_set.values()])
                map_embeddings.append(temp_map_embeddings)
                map_local_embeddings.append(temp_map_local_embeddings)
                map_positions.append(temp_map_positions)
            # with open(os.path.join(save_path, 'map_embeddings.pickle'), 'wb') as handle:
            #     pickle.dump(map_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(os.path.join(save_path, 'map_local_embeddings.pickle'), 'wb') as handle:
            #     pickle.dump(map_local_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # # UNCOMMENT BLOCK WHEN LOADING FROM PICKLE
            # for map_set in map_sets:
            #     temp_map_positions = np.array([(x['northing'], x['easting']) for x in map_set.values()])
            #     map_positions.append(temp_map_positions)
            # with open(os.path.join(save_path, 'map_embeddings.pickle'), 'rb') as handle:
            #     map_embeddings = pickle.load(handle)
            # with open(os.path.join(save_path, 'map_local_embeddings.pickle'), 'rb') as handle:
            #     map_local_embeddings = pickle.load(handle)

            # ### TEMP JUST FOR PRE-COMPUTING EMBEDDINGS ###
            # continue
            # ### TEMP JUST FOR PRE-COMPUTING EMBEDDINGS ###
            
            global_metrics_splits, local_metrics_splits = {}, {}
            for i in range(len(map_sets)):
                for j in range(len(query_sets)):
                    if (i == j and skip_same_run) or map_embeddings[i] is None or query_embeddings[j] is None:
                        continue 
                    print(f'DATABASE {i} -- QUERY {j}:', flush=True)
                    temp_global_metrics, temp_local_metrics = self.evaluate_split(
                        i,
                        j,
                        query_embeddings[j],
                        map_embeddings[i],
                        query_local_embeddings[j],
                        map_local_embeddings[i],
                        map_positions[i],
                        query_sets[j],
                        map_sets[i],
                        d_thresh,
                    )
                    global_metrics_splits[f'db{i}-q{j}'] = temp_global_metrics
                    local_metrics_splits[f'db{i}-q{j}'] = temp_local_metrics
            # Compute average for location
            global_metrics_splits['average'] = average_nested_dict(global_metrics_splits)
            global_metrics[location] = global_metrics_splits
            if not self.only_global:
                local_metrics_splits['average'] = average_nested_dict(local_metrics_splits)
                local_metrics[location] = local_metrics_splits

        # Compute average for dataset
        global_metrics['average'] = {'average': average_nested_dict({loc: inner_dict['average'] for loc, inner_dict in global_metrics.items()})}
        if not self.only_global:
            local_metrics['average'] = {'average': average_nested_dict({loc: inner_dict['average'] for loc, inner_dict in local_metrics.items()})}
        return global_metrics, local_metrics

    def evaluate_split(
        self,
        m,
        n,
        query_embeddings,
        map_embeddings,
        query_local_embeddings,
        map_local_embeddings,
        map_positions,
        query_set,
        map_set,
        d_thresh,
        *args,
        **kwargs,
    ):
        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        if self.only_global:
            metrics = {}
        else:
            metrics = {eval_mode: {'rre': [], 'rte': [], 'repeatability': [],
                                'success': [], 'success_inliers': [], 'failure_inliers': [],
                                'rre_refined': [], 'rte_refined': [], 'success_refined': [],
                                'success_inliers_refined': [], 'repeatability_refined': [],
                                'failure_inliers_refined': [], 't_ransac': []}
                       for eval_mode in ['Initial', 'Re-Ranked']}

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {}
        intermediate_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        intermediate_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        intermediate_metrics['RR'] = {r: [] for r in self.radius}
        intermediate_metrics['RR_rr'] = {r: [] for r in self.radius}
        intermediate_metrics['t_rr'] = []

        num_evaluated = 0

        for query_ndx in tqdm.tqdm(query_indexes, desc='Evaluating'):
            ######################## DEBUGGING #################################
            if DEBUG and query_ndx >= DEBUG_MAX_QUERIES:
                break
            ######################## DEBUGGING #################################

            # TEMP FIX FOR MISSING QUERY FILES
            if query_local_embeddings[query_ndx] is None:
                print(f'Skipping {query_ndx}', flush=True)
                continue
            
            # Check if the query element has a true match within each radius
            query_metadata = query_set[query_ndx]  # {'query': path, 'northing': , 'easting': , 'pose': }
            query_position = np.array((query_metadata['northing'], query_metadata['easting']))
            query_pose = query_metadata['pose']
            if m in query_metadata:  # old tuples store true neighbours directly
                true_neighbors = query_metadata[m]
                if len(true_neighbors) == 0:
                    continue
            else:  # expected that new tuples filter queries, but do here just in case
                min_neighbor_dist = np.linalg.norm(query_position - map_positions, axis=-1).min()
                if min_neighbor_dist > min(self.radius):
                    continue
            num_evaluated += 1
            
            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            # embed_dist = torch.cdist(torch.tensor(query_embedding).unsqueeze(0).unsqueeze(0), torch.tensor(map_embeddings).unsqueeze(0)).squeeze().numpy()
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # PLACE RECOGNITION EVALUATION
            # Euclidean distance between the query and nn
            # Here we use non-icp refined poses, but for the global descriptor it's fine
            delta = query_position - map_positions[nn_ndx]       # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array

            # re_rank = True
            if d_thresh > 0:
                tick = time.perf_counter()
                if DEBUG_CSWP_FAILURE:
                    topk_rerank_inds = copy.deepcopy(nn_ndx)
                else:
                    # fitness_list = []
                    # fitness_list = np.zeros(len(nn_ndx))
                    topk = min(self.k,len(nn_ndx))
                    fitness_list = np.zeros(topk)
                    for k in range(topk):
                        k_id = nn_ndx[k]
                        # TEMP FIX FOR MISSING MAP FILES
                        if map_local_embeddings[k_id] is None:
                            continue
                        conf_val = sgv_fn(
                            query_local_embeddings[query_ndx],
                            map_local_embeddings[k_id],
                            d_thresh=d_thresh,
                            max_points=self.num_points,
                        )
                        # fitness_list.append(conf_val)
                        fitness_list[k] = conf_val
                    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                    # topk_rerank = np.flip(np.asarray(fitness_list).argsort())
                    # topk_rerank_inds = nn_ndx[topk_rerank]
                    topk_rerank = np.flip(np.asarray(fitness_list).argsort())
                    topk_rerank_inds = copy.deepcopy(nn_ndx)
                    topk_rerank_inds[:topk] = nn_ndx[topk_rerank]
                t_rerank = time.perf_counter() - tick
                intermediate_metrics['t_rr'].append(t_rerank)

                delta_rerank = query_position - map_positions[topk_rerank_inds]
                euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
            else:
                euclid_dist_rr = euclid_dist
                intermediate_metrics['t_rr'].append(0)

            # Count true positives for different radius and NN number
            intermediate_metrics['tp'] = {r: [intermediate_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            intermediate_metrics['tp_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            intermediate_metrics['RR'] = {r: intermediate_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in self.radius}
            intermediate_metrics['RR_rr'] = {r: intermediate_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
            if self.only_global:
                continue

            # METRIC LOCALIZATION EVALUATION
            # Do the evaluation only if the nn pose is within distance threshold
            # Otherwise the overlap is too small to get reasonable results
            # (evaluation continues only if standard AND re-ranked nn pose is within thresh meters,
            #  to ensure both are evaluated on the same set of queries)
            if euclid_dist[0] > self.local_max_eval_threshold or euclid_dist_rr[0] > self.local_max_eval_threshold:
                continue

            for eval_mode in ['Initial', 'Re-Ranked']:
                # Get the first match and compute local stats only for the best match
                if eval_mode == 'Initial':
                    nn_idx = nn_ndx[0]
                elif eval_mode == 'Re-Ranked':
                    nn_idx = topk_rerank_inds[0]

                tick = time.perf_counter()
                if DEBUG_CSWP_FAILURE:
                    T_estimated, inliers = np.eye(4), 0
                else:
                    # Ransac alignment
                    T_estimated, inliers, _ = self.ransac_fn(query_local_embeddings[query_ndx],
                                                            map_local_embeddings[nn_idx])
                t_ransac = time.perf_counter() - tick
                nn_pose = map_set[nn_idx]['pose']
                if self.dataset_type == 'mulran':
                    T_gt = mulran_relative_pose(query_pose, nn_pose)
                else:
                    T_gt = relative_pose(query_pose, nn_pose)

                # Refine the pose using ICP
                if (not self.icp_refine) or DEBUG_CSWP_FAILURE:
                    T_refined = T_gt
                else:
                    query_filepath = os.path.join(self.dataset_root, query_set[query_ndx]['query'])
                    query_pc = self.pc_loader(query_filepath)
                    map_filepath = os.path.join(self.dataset_root, map_set[nn_idx]['query'])
                    map_pc = self.pc_loader(map_filepath)
                    # # TODO: Can remove this preprocessing -- already done in `pc_loader`
                    # if self.dataset_type in ['mulran', 'kitti', 'kitti360']:
                    #     query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                    #                                      min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                    #     map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                    #                                      min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                    # elif self.dataset_type in ['southbay', 'alita']:
                    #     # -1.6 removes most of the ground plane
                    #     query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                    #                                      min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                    #     map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                    #                                    min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                    # elif 'wild' in dataset_type:
                    #     # TODO
                    #     query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True)
                    #     map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True)
                    # else:
                    #     raise NotImplementedError(f"Unknown dataset type: {self.dataset_type}")

                    match self.dataset_type:
                        case 'mulran':
                            T_refined, _, _ = two_stage_icp(query_pc, map_pc, T_gt)
                        case _:
                            T_refined, _, _ = icp(query_pc, map_pc, T_gt)

                # Compute repeatability using refined pose (IGNORED TO SAVE TIME)
                # kp1 = query_local_embeddings[query_ndx]['keypoints']
                # kp2 = map_local_embeddings[nn_idx]['keypoints']
                # metrics[eval_mode]['repeatability'].append(calculate_repeatability(kp1, kp2, T_gt, threshold=self.repeat_dist_th))
                # metrics[eval_mode]['repeatability_refined'].append(calculate_repeatability(kp1, kp2, T_refined, threshold=self.repeat_dist_th))
                metrics[eval_mode]['repeatability'].append(0.0)
                metrics[eval_mode]['repeatability_refined'].append(0.0)

                # calc errors
                rte = np.linalg.norm(T_estimated[:3, 3] - T_gt[:3, 3])
                cos_rre = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_gt[:3, :3]) - 1.) / 2.
                rre = np.arccos(np.clip(cos_rre, a_min=-1., a_max=1.)) * 180. / np.pi

                metrics[eval_mode]['t_ransac'].append(t_ransac)    # RANSAC time

                # 2 meters and 5 degrees threshold for successful registration
                if rte > 2.0 or rre > 5.0:
                    metrics[eval_mode]['success'].append(0.)
                    metrics[eval_mode]['failure_inliers'].append(inliers)
                else:
                    metrics[eval_mode]['success'].append(1.)
                    metrics[eval_mode]['success_inliers'].append(inliers)
                metrics[eval_mode]['rte'].append(rte)
                metrics[eval_mode]['rre'].append(rre)

                if self.icp_refine:
                    # calc errors using refined pose
                    rte_refined = np.linalg.norm(T_estimated[:3, 3] - T_refined[:3, 3])
                    cos_rre_refined = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_refined[:3, :3]) - 1.) / 2.
                    rre_refined = np.arccos(np.clip(cos_rre_refined, a_min=-1., a_max=1.)) * 180. / np.pi

                    # 2 meters and 5 degrees threshold for successful registration
                    if rte_refined > 2.0 or rre_refined > 5.0:
                        metrics[eval_mode]['success_refined'].append(0.)
                        metrics[eval_mode]['failure_inliers_refined'].append(inliers)
                    else:
                        metrics[eval_mode]['success_refined'].append(1.)
                        metrics[eval_mode]['success_inliers_refined'].append(inliers)
                    metrics[eval_mode]['rte_refined'].append(rte_refined)
                    metrics[eval_mode]['rre_refined'].append(rre_refined)

        # Calculate mean metrics
        global_metrics['recall'] = {r: [intermediate_metrics['tp'][r][nn] / num_evaluated for nn in range(self.k)] for r in self.radius}
        global_metrics['recall_rr'] = {r: [intermediate_metrics['tp_rr'][r][nn] / num_evaluated for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(intermediate_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(intermediate_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_rr'] = {r: np.mean(np.asarray(intermediate_metrics['t_rr'])) for r in self.radius}  # duplicating for each radius so `average_nested_dict` doesn't break

        local_metrics = {}
        if not self.only_global:
            # Calculate mean values of local descriptor metrics
            for eval_mode in ['Initial', 'Re-Ranked']:
                local_metrics[eval_mode] = {}
                for metric in metrics[eval_mode]:
                    m_l = metrics[eval_mode][metric]
                    if len(m_l) == 0:
                        local_metrics[eval_mode][metric] = 0.
                    else:
                        if metric == 't_ransac':
                            local_metrics[eval_mode]["t_ransac_sd"] = np.std(m_l)
                        local_metrics[eval_mode][metric] = np.mean(m_l)

        return global_metrics, local_metrics

    def ransac_fn(self, query_keypoints, candidate_keypoints):
        """

        Returns fitness score and estimated transforms
        Estimation using Open3d 6dof ransac based on feature matching.
        """
        kp1 = query_keypoints['keypoints']
        kp2 = candidate_keypoints['keypoints']
        ransac_result = get_ransac_result(query_keypoints['features'], candidate_keypoints['features'],
                                          kp1, kp2)
        return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness

    def compute_embeddings(self, eval_subset: dict, model):
        self.model2eval((model,))
        global_embeddings = None
        local_embeddings = []
        for ndx, e in enumerate(tqdm.tqdm(eval_subset.values(), desc='Computing embeddings')):
            scan_filepath = os.path.join(self.dataset_root, e['query'])
            # assert os.path.exists(scan_filepath)
            if not os.path.exists(scan_filepath):
                print(f'Missing file for idx {ndx}')
                # global_embeddings[ndx] = None
                local_embeddings.append(None)
                continue
            pc = self.pc_loader(scan_filepath)

            oo = np.ones(len(pc)).reshape((-1,1))
            xyzi = np.hstack((pc, oo)).astype(np.float32)
            hash_vals, coords, indices, input, points = make_sparse_tensor(
                xyzi,
                voxel_size=self.voxel_size,
                num_points=self.num_points,
                return_points=True,
                return_hash=True,
            )
            coords = coords.astype(np.int32)
            input = input.cuda()
            points = points[:,:3]

            # output_desc, output_feats, xc = model(input)
            global_embedding, keypoints, key_embeddings = self.compute_embedding(input, model, coords, points, is_dense=True)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding
            local_embeddings.append({'keypoints': keypoints, 'features': key_embeddings})

        return global_embeddings, local_embeddings

    def compute_embedding(self, pc, model, coords, points, is_dense=False):
        """
        Returns global embedding (np.array) as well as keypoints and corresponding descriptors (torch.tensors)
        """
        # output_desc, output_feats, xc = model(pc)
        output_desc, output_feats = model(pc)

        if is_dense:
            output_features = output_feats[0].cpu().detach().numpy()
            output_points = points
        else:
            raise NotImplementedError
            # xc_coords = xc.C
            # xc_feat = xc.F.cpu().detach().numpy()
            # _, xc_counts = torch.unique(xc_coords[:, -1], return_counts=True)
            # y = torch.split(xc_coords, list(xc_counts))
            # assert len(y) == 1
            # xc_coords = xc_coords.cpu().detach().numpy()
            # xc_coords= xc_coords[:,:3]
            # len_coords = len(coords)
            # all_coords = np.vstack((coords,xc_coords))
            # all_hash_vals = ravel_hash(all_coords)

            # dense_hash = all_hash_vals[:len_coords]
            # coarse_hash = all_hash_vals[len_coords:]

            # sorter = np.argsort(dense_hash)
            # xc_hash_inds = sorter[np.searchsorted(dense_hash, coarse_hash, sorter=sorter)]

            # unique_xc_hash_inds, unique_xc_hash_inds_indices = np.unique(xc_hash_inds,
            #                                     return_index=True)

            # output_points = points[unique_xc_hash_inds]
            # output_features = xc_feat[unique_xc_hash_inds_indices]

        global_descriptor = output_desc.cpu().detach().numpy()
        global_embedding = np.reshape(global_descriptor, (1, -1))

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        return global_embedding, torch.tensor(output_points, dtype=torch.float), torch.tensor(output_features, dtype=torch.float)


    # def print_results(self, global_metrics, metrics):
    #     # Global descriptor results are saved with the last n_k entry
    #     print('\n','Initial Retrieval:')
    #     recall = global_metrics['recall']
    #     for r in recall:
    #         print(f"Radius: {r} [m] : ")
    #         print(f"Recall@N : ", end='')
    #         for x in recall[r]:
    #             print("{:0.1f}, ".format(x*100.0), end='')
    #         print("")
    #         print('MRR: {:0.1f}'.format(global_metrics['MRR'][r]*100.0))
        
    #     print('\n','Re-Ranking:')
    #     recall_rr = global_metrics['recall_rr']
    #     for r_rr in recall_rr:
    #         print(f"Radius: {r_rr} [m] : ")
    #         print(f"Recall@N : ", end='')
    #         for x in recall_rr[r_rr]:
    #             print("{:0.1f}, ".format(x*100.0), end='')
    #         print("")
    #         print('MRR: {:0.1f}'.format(global_metrics['MRR_rr'][r_rr]*100.0))
    #     print('Re-Ranking Time: {:0.3f}'.format(1000.0 *global_metrics['mean_t_RR']))

    #     print('\n','Metric Localization:')
    #     for eval_mode in ['Initial', 'Re-Ranked']:
    #         if eval_mode not in metrics:
    #             break
    #         print('#keypoints: {}'.format(eval_mode))
    #         for s in metrics[eval_mode]:
    #             print(f"{s}: {metrics[eval_mode][s]:0.3f}")
    #         print('')

    def print_results(
        self,
        global_metrics: dict,
        local_metrics: dict,
        icp_refine=False,
        print_false_positives=False,
    ):
        msg = 'Eval Results'
        for database_name in global_metrics.keys():
            msg += '\nDataset: {}'.format(database_name)
            for split in global_metrics[database_name].keys():
                msg += '\n  Split: {}'.format(split)
                msg += '\n    Initial Retrieval:'
                recall = global_metrics[database_name][split]['recall']
                for radius in recall.keys():
                    msg += f"\n      Radius: {radius} [m]:\n        Recall@N:\n          "
                    for ii, x in enumerate(recall[radius]):
                            msg += f"{x:0.4f},"
                            # if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                            #     msg += "\n                  "
                    # msg += "\n        Recall@1%: {:0.4f}".format(global_metrics[database_name][split]['recall@1%'][radius])
                    msg += '\n        MRR: {:0.4f}'.format(global_metrics[database_name][split]['MRR'][radius])

                msg += '\n    Re-Ranking (sgv):'
                recall_rr = global_metrics[database_name][split]['recall_rr']
                for radius_rr in recall_rr.keys():
                    msg += f"\n      Radius: {radius_rr} [m]:\n        Recall@N:\n          "
                    for ii, x in enumerate(recall_rr[radius_rr]):
                        msg += f"{x:0.4f},"
                        # if (ii+1) % 5 == 0 and (ii+1) < len(recall[radius]):
                        #     msg += "\n                  "
                    # msg += "\n        Recall@1%: {:0.4f}".format(global_metrics[database_name][split]['recall@1%_rr'][radius_rr])
                    msg += '\n        MRR: {:0.4f}'.format(global_metrics[database_name][split]['MRR_rr'][radius_rr])
                msg += '\n        Re-Ranking Time: {:0.4f} [ms]'.format(1000.0*global_metrics[database_name][split]['mean_t_rr'][radius_rr])
                if (print_false_positives and 'rr_failures' in global_metrics[database_name][split]
                    and len(global_metrics[database_name][split]['rr_failures']) > 0):
                    msg += '\n        Re-Ranking Failures (query, nn, nn_rerank, nn_dist, nn_rerank_dist):\n          '
                    msg += '\n          '.join([', '.join(x) for x in global_metrics[database_name][split]['rr_failures']])

                if len(local_metrics) == 0:
                    msg += '\n'
                    continue
                msg += '\n    Metric Localization:'
                for eval_mode in ['Initial', 'Re-Ranked']:
                    if eval_mode not in local_metrics[database_name][split]:
                        continue
                    msg += f'\n      Eval Mode: {eval_mode}'
                    for metric in local_metrics[database_name][split][eval_mode]:
                        if '_refined' in metric and not icp_refine:
                            continue
                        if 'failure_query_pos_ndx' in metric:
                            if not print_false_positives:
                                continue
                            msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]}"
                        else:
                            msg += f"\n      {metric}: {local_metrics[database_name][split][eval_mode][metric]:0.3f}"
                msg += '\n'
            msg += '\n'
        print(msg, flush=True)


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000, random_seed=42):
    o3d.utility.random.seed(random_seed)
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    if not isinstance(kp1, np.ndarray):
        pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
        pcd_coord2 = make_open3d_point_cloud(kp2.numpy())
    else:
        pcd_coord1 = make_open3d_point_cloud(kp1)
        pcd_coord2 = make_open3d_point_cloud(kp2)

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result


def calculate_repeatability(kp1, kp2, T_gt, threshold: float):
    # Transform the source point cloud to the same position as the target cloud
    kp1_pos_trans = apply_transform(kp1, torch.tensor(T_gt, dtype=torch.float))
    dist = torch.cdist(kp1_pos_trans, kp2)      # (n_keypoints1, n_keypoints2) tensor

    # *** COMPUTE REPEATABILITY ***
    # Match keypoints from the first cloud with closests keypoints in the second cloud
    min_dist, _ = torch.min(dist, dim=1)
    # Repeatability with a distance threshold th
    return torch.mean((min_dist <= threshold).float()).item()

def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_hash: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    hash_vals, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    if return_hash: outputs = [hash_vals, coords]
    else: outputs = [coords]

    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x -= np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def make_sparse_tensor(lidar_pc, voxel_size=0.05, num_points=150000, return_points=False, return_hash=False):
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords_min = coords.min(0, keepdims=1)
    coords -= coords_min
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    hash_vals, _, indices = sparse_quantize(coords, return_index=True, return_hash=return_hash)
    if len(indices) > num_points:
          indices = np.random.choice(indices, num_points, replace=False)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()
    if return_points:
        return hash_vals, coords, indices, inputs , feats
    else:
        return inputs

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    # parser.add_argument('--dataset_root', type=str, required=False, default='', help='Path to the dataset root')
    # parser.add_argument('--dataset_type', type=str, required=False, default='kitti360', choices=['mulran', 'southbay', 'kitti', 'alita', 'kitti360'])
    # parser.add_argument('--eval_set', type=str, required=False, default='kitti_00_eval.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    # parser.add_argument('--radius', type=float, nargs='+', default=[5, 20], help='True Positive thresholds in meters')
    # parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    # parser.add_argument('--weights', type=str, default='/logg3d.pth')
    # parser.add_argument('--model', type=str, default='logg3d1k', choices=['logg3d', 'logg3d1k'])
    # parser.add_argument('--d_thresh', type=float, default=0.4, help='Dist thresholds in meters')
    # parser.add_argument('--n_topk', type=int, default=20, help='Num retrievals to consider per query')
    # parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    # parser.set_defaults(icp_refine=True)
    # parser.add_argument('--local_max_eval_threshold', type=float, default=np.inf,
    #                     help='Maxmimum allowable retrieval distance to continue with metric localisation evaluation')

    # args = parser.parse_args()
    # if args.dataset_type == 'kitti':
    #     args.eval_set = 'kitti_00_eval.pickle'
    # elif args.dataset_type == 'mulran':
    #     if args.mulran_sequence == 'sejong':
    #         args.eval_set = 'test_Sejong_01_Sejong_02.pickle'
    #     else:
    #         args.eval_set = 'test_DCC_01_DCC_02_10.0_5.pickle'
    # elif args.dataset_type == 'southbay':
    #     args.eval_set = 'test_SunnyvaleBigloop_1.0_5.pickle'
    # elif args.dataset_type == 'alita':
    #     args.eval_set = 'test_val_5_0.01_5.pickle'
    # elif args.dataset_type == 'kitti360':
    #     args.eval_set = 'kitti360_09_3.0_eval.pickle'

    # print(f'Dataset root: {args.dataset_root}')
    # print(f'Dataset type: {args.dataset_type}')
    # print(f'Evaluation set: {args.eval_set}')
    # print(f'Radius: {args.radius} [m]')
    # print(f'd_thresh: {args.d_thresh} [m]')
    # print(f'n_topk: {args.n_topk} ')
    # print(f'')
    # print('')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    # Set random seed for (mostly) deterministic behaviour
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    o3d.utility.random.seed(seed)

    cfg = get_config_eval()
    assert cfg.eval_batch_size == 1, "Batch size > 1 not supported"

    if 'General' in cfg.eval_dataset:  # CS wildplaces (for now)
        match cfg.eval_dataset_name:
            case 'cs_campus3d':
                root_dir = cfg.cs_campus3d_dir
            case 'cs_wildplaces':
                root_dir = cfg.cs_wildplaces_dir
            case 'm3gawild':
                root_dir = cfg.cs_wildplaces_dir
            case 'm3gawild_airborne':
                root_dir = cfg.cs_wildplaces_dir
            case 'mulran':
                root_dir = cfg.mulran_dir
            case _:
                raise NotImplementedError(f'{cfg.eval_dataset_name} not implemented')
    elif cfg.eval_dataset == 'WildPlacesDataset':
        root_dir = cfg.wildplaces_dir
    else:
        raise NotImplementedError(f'{cfg.eval_dataset} not implemented')

    model = get_pipeline(cfg.eval_pipeline, cfg)
    if cfg.checkpoint_name is not None:
        save_path = os.path.join(os.path.dirname(__file__), '../../')
        save_path = os.path.join(save_path, cfg.checkpoint_name)
        assert os.path.exists(save_path), 'Cannot open network weights: {}'.format(save_path)
        print('Loading weights: {}'.format(save_path))
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    # evaluator = MetLocEvaluator(args.dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius, k=args.n_topk,
    evaluator = MetLocEvaluator(root_dir, cfg.eval_dataset_name, device, radius=cfg.radius, k=cfg.n_topk,
                                n_samples=cfg.n_samples, voxel_size=cfg.voxel_size, icp_refine=cfg.icp_refine,
                                num_points=cfg.num_points, local_max_eval_threshold=cfg.local_max_eval_threshold,
                                sequence_idx=cfg.sequence_idx)
    global_metrics, local_metrics = evaluator.evaluate(model, d_thresh=cfg.d_thresh, only_global=cfg.only_global)
    evaluator.print_results(global_metrics, local_metrics, icp_refine=cfg.icp_refine)
