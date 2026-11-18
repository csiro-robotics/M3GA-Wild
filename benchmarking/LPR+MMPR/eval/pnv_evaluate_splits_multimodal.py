# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# Adapted to process samples in batches by Ethan Griffiths (QUT & CSIRO Data61).
# Adapted to compare multimodal features by Ethan Griffiths (QUT & CSIRO Data61).
# Adapted to optionally limit GPS search radius

import os
import pickle
import argparse
import logging

import torch
import tqdm
import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

from models.model_factory import model_factory
from misc.logger import create_logger
from misc.utils import TrainingParams
from misc.torch_utils import set_seed
from eval.utils import get_query_database_splits
from eval.pnv_evaluate import (
    get_latent_vectors,
    save_embeddings_to_file,
    load_embeddings_from_file,
)

args = None
NUM_NEIGHBORS = 25  # default 25

def evaluate(model, device, params: TrainingParams, log: bool = False,
             model_name: str = 'model', show_progress: bool = False, 
             save_embeddings: bool = False, load_embeddings: bool = False):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)
    stats = {}
    ave_recall, ave_recall_image, ave_recall_cloud = [], [], []
    ave_one_percent_recall, ave_one_percent_recall_image, ave_one_percent_recall_cloud = [], [], []
    ave_mrr, ave_mrr_image, ave_mrr_cloud = [], [], []
    ave_top1_geo_dist, ave_top1_geo_dist_image, ave_top1_geo_dist_cloud = [], [], []
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        if any([name in params.dataset_name for name in ['AboveUnder', 'CSWildPlaces', 'M3GAWild']]):
            if "pickles/" in database_file:  # CS-WildPlaces
                location_name = database_file.split('/')[-1].split('_')[1]
                temp = query_file.split('/')[-1].split('_')[1]
            else:
                location_name = database_file.split('_')[1]
                temp = query_file.split('_')[1]
        else:
            location_name = database_file.split('_')[0]
            temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        if show_progress:
            logging.info(f'Evaluating: {location_name}')
        temp = evaluate_dataset(model, device, params, database_sets, query_sets, location_name,
                                log=log, model_name=model_name, show_progress=show_progress,
                                save_embeddings=save_embeddings, load_embeddings=load_embeddings)
        stats[location_name] = temp
        ave_one_percent_recall.append(temp['average']['ave_one_percent_recall'])
        ave_recall.append(temp['average']['ave_recall'])
        ave_mrr.append(temp['average']['ave_mrr'])
        ave_top1_geo_dist.append(temp['average']['ave_top1_geo_dist'])
        ave_one_percent_recall_image.append(temp['average']['ave_one_percent_recall_image'])
        ave_recall_image.append(temp['average']['ave_recall_image'])
        ave_mrr_image.append(temp['average']['ave_mrr_image'])
        ave_top1_geo_dist_image.append(temp['average']['ave_top1_geo_dist_image'])
        ave_one_percent_recall_cloud.append(temp['average']['ave_one_percent_recall_cloud'])
        ave_recall_cloud.append(temp['average']['ave_recall_cloud'])
        ave_mrr_cloud.append(temp['average']['ave_mrr_cloud'])
        ave_top1_geo_dist_cloud.append(temp['average']['ave_top1_geo_dist_cloud'])
        
    # Compute average stats
    stats['average'] = {'average': {
        'ave_one_percent_recall': np.mean(ave_one_percent_recall),
        'ave_recall': np.mean(ave_recall, axis=0),
        'ave_mrr': np.mean(ave_mrr),
        'ave_top1_geo_dist': np.mean(ave_top1_geo_dist),
        'ave_one_percent_recall_image': np.mean(ave_one_percent_recall_image),
        'ave_recall_image': np.mean(ave_recall_image, axis=0),
        'ave_mrr_image': np.mean(ave_mrr_image),
        'ave_top1_geo_dist_image': np.mean(ave_top1_geo_dist_image),
        'ave_one_percent_recall_cloud': np.mean(ave_one_percent_recall_cloud),
        'ave_recall_cloud': np.mean(ave_recall_cloud, axis=0),
        'ave_mrr_cloud': np.mean(ave_mrr_cloud),
        'ave_top1_geo_dist_cloud': np.mean(ave_top1_geo_dist_cloud),
    }}
    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets: list[dict], query_sets: list[dict],
                     location_name: str, log: bool = False, model_name: str = 'model',
                     show_progress: bool = False, save_embeddings: bool = False,
                     load_embeddings: bool = False):
    # Run evaluation on a single dataset
    stats = {}
    count = 0
    recall = np.zeros(NUM_NEIGHBORS)
    one_percent_recall = []
    mrr = []
    top1_geo_dist = []
    recall_image = np.zeros(NUM_NEIGHBORS)
    one_percent_recall_image = []
    mrr_image = []
    top1_geo_dist_image = []
    recall_cloud = np.zeros(NUM_NEIGHBORS)
    one_percent_recall_cloud = []
    mrr_cloud = []
    top1_geo_dist_cloud = []

    database_embeddings = []
    query_embeddings = []
    database_positions = []
    query_positions = []

    model.eval()

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} database embeddings')
    for ii, data_set in enumerate(database_sets):
        temp_embeddings, temp_positions = None, None
        if len(data_set) > 0:
            # Create array of coordinates of all db elements
            temp_positions = np.array([(sample_details['northing'], sample_details['easting']) for sample_details in data_set.values()])
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{ii}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'database_{ii}')
        database_embeddings.append(temp_embeddings)
        database_positions.append(temp_positions)

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} query embeddings')
    for jj, data_set in enumerate(query_sets):
        temp_embeddings, temp_positions = None, None
        if len(data_set) > 0:
            # Create array of coordinates of all query elements
            temp_positions = np.array([(sample_details['northing'], sample_details['easting']) for sample_details in data_set.values()])
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{jj}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'query_{jj}')
        query_embeddings.append(temp_embeddings)
        query_positions.append(temp_positions)

    if show_progress:
        logging.info('Running evaluation')
    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            if (i == j and params.skip_same_run) or database_embeddings[i] is None or query_embeddings[j] is None:
                continue
            if 'CSCampus3D' in params.dataset_name:
                # For Campus3D, we report on the aerial-only database, which is idx 1
                if i != 1:
                    continue
                split_name = os.path.split(os.path.split(database_sets[i][0]['query'])[0])[0] + f'_idx{i}'
            else:
                split_name = os.path.split(os.path.split(query_sets[j][0]['query'])[0])[0]
            pair_recall, pair_opr, pair_mrr, pair_top1_geo_dist = get_recall(
                i, j, database_embeddings, query_embeddings, database_sets, query_sets,
                database_positions, query_positions, location_name=location_name, log=log,
                model_name=model_name, embedding_type='embeddings',
            )
            pair_recall_image, pair_opr_image, pair_mrr_image, pair_top1_geo_dist_image = get_recall(
                i, j, database_embeddings, query_embeddings, database_sets, query_sets,
                database_positions, query_positions, location_name=location_name, log=log,
                model_name=model_name, embedding_type='image_embeddings',
            )
            pair_recall_cloud, pair_opr_cloud, pair_mrr_cloud, pair_top1_geo_dist_cloud = get_recall(
                i, j, database_embeddings, query_embeddings, database_sets, query_sets,
                database_positions, query_positions, location_name=location_name, log=log,
                model_name=model_name, embedding_type='cloud_embeddings',
            )
            if log:
                # Find lidar failure cases where image succeeded
                log_path = os.path.join('./log', location_name, model_name)
                image_succ_file = os.path.join(log_path, f"db{i}_q{j}_log_image_successes_vs_cloud.txt")
                image_fp_file_name = os.path.join(log_path, f"db{i}_q{j}_log_image_embeddings_fp.txt")
                cloud_fp_file_name = os.path.join(log_path, f"db{i}_q{j}_log_cloud_embeddings_fp.txt")
                df_image_fp = pd.read_csv(image_fp_file_name, index_col='query_idx')
                df_cloud_fp = pd.read_csv(cloud_fp_file_name, index_col='query_idx')
                image_fp_indices = df_image_fp.index.to_numpy()
                cloud_fp_indices = df_cloud_fp.index.to_numpy()
                image_success_indices = np.setdiff1d(cloud_fp_indices, image_fp_indices)
                image_search_file_name = os.path.join(log_path, f"db{i}_q{j}_log_image_embeddings_search_results.txt")
                cloud_search_file_name = os.path.join(log_path, f"db{i}_q{j}_log_cloud_embeddings_search_results.txt")
                df_image_search = pd.read_csv(image_search_file_name, index_col='query_idx')
                df_cloud_search = pd.read_csv(cloud_search_file_name, index_col='query_idx')
                with open(image_succ_file, 'w') as f:
                    s = 'query_idx,query_image,tp_image,tp_geo_dist,query_cloud,fp_cloud,fp_geo_dist\n'
                    for succ_idx in image_success_indices:
                        s += '{},{},{},{:0.2f},{},{},{:0.2f}\n'.format(
                            succ_idx,
                            query_sets[j][succ_idx]['image'],
                            df_image_search.loc[succ_idx]['top1'],
                            df_image_search.loc[succ_idx]['top1_geo_dist'],
                            query_sets[j][succ_idx]['query'],
                            df_cloud_search.loc[succ_idx]['top1'],
                            df_cloud_search.loc[succ_idx]['top1_geo_dist'],
                        )
                    f.write(s)
            count += 1
            recall += np.array(pair_recall)
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)
            top1_geo_dist.append(pair_top1_geo_dist)
            recall_image += np.array(pair_recall_image)
            one_percent_recall_image.append(pair_opr_image)
            mrr_image.append(pair_mrr_image)
            top1_geo_dist_image.append(pair_top1_geo_dist_image)
            recall_cloud += np.array(pair_recall_cloud)
            one_percent_recall_cloud.append(pair_opr_cloud)
            mrr_cloud.append(pair_mrr_cloud)
            top1_geo_dist_cloud.append(pair_top1_geo_dist_cloud)
            
            # Report per-split metrics
            stats[split_name] = {
                'ave_one_percent_recall': pair_opr,
                'ave_recall': pair_recall,
                'ave_mrr': pair_mrr,
                'ave_top1_geo_dist': pair_top1_geo_dist,
                'ave_one_percent_recall_image': pair_opr_image,
                'ave_recall_image': pair_recall_image,
                'ave_mrr_image': pair_mrr_image,
                'ave_top1_geo_dist_image': pair_top1_geo_dist_image,
                'ave_one_percent_recall_cloud': pair_opr_cloud,
                'ave_recall_cloud': pair_recall_cloud,
                'ave_mrr_cloud': pair_mrr_cloud,
                'ave_top1_geo_dist_cloud': pair_top1_geo_dist_cloud,
            }
    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    ave_mrr = np.mean(mrr)
    ave_top1_geo_dist = np.mean(top1_geo_dist)
    ave_recall_image = recall_image / count
    ave_one_percent_recall_image = np.mean(one_percent_recall_image)
    ave_mrr_image = np.mean(mrr_image)
    ave_top1_geo_dist_image = np.mean(top1_geo_dist_image)
    ave_recall_cloud = recall_cloud / count
    ave_one_percent_recall_cloud = np.mean(one_percent_recall_cloud)
    ave_mrr_cloud = np.mean(mrr_cloud)
    ave_top1_geo_dist_cloud = np.mean(top1_geo_dist_cloud)
    stats['average'] = {
        'ave_one_percent_recall': ave_one_percent_recall,
        'ave_recall': ave_recall,
        'ave_mrr': ave_mrr,
        'ave_top1_geo_dist': ave_top1_geo_dist,
        'ave_one_percent_recall_image': ave_one_percent_recall_image,
        'ave_recall_image': ave_recall_image,
        'ave_mrr_image': ave_mrr_image,
        'ave_top1_geo_dist_image': ave_top1_geo_dist_image,
        'ave_one_percent_recall_cloud': ave_one_percent_recall_cloud,
        'ave_recall_cloud': ave_recall_cloud,
        'ave_mrr_cloud': ave_mrr_cloud,
        'ave_top1_geo_dist_cloud': ave_top1_geo_dist_cloud,
    }
    return stats


def get_recall(
    m: int,
    n: int,
    database_vectors: dict,
    query_vectors: dict,
    database_sets: list[dict],
    query_sets: list[dict],
    database_positions: list[npt.NDArray],
    query_positions: list[npt.NDArray],
    location_name: str,
    log=False,
    model_name: str = 'model',
    embedding_type: str = 'embeddings',
):
    assert embedding_type in ('embeddings', 'image_embeddings', 'cloud_embeddings')
    gps_radius = 0.0
    if args is not None:
        gps_radius = args.gps_radius
    database_output = database_vectors[m][embedding_type]
    queries_output = query_vectors[n][embedding_type]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    db_faiss_index = faiss.IndexFlatL2(database_output.shape[1])
    db_faiss_index.add(database_output)
    faiss_distances, faiss_predictions = db_faiss_index.search(queries_output, k=NUM_NEIGHBORS)
    embedding_distances = cdist(queries_output, database_output, metric='euclidean')

    # Store geographic distances between all query/db pairs
    geographic_distances = cdist(query_positions[n], database_positions[m], metric='euclidean')
    valid_gps_mask = None
    if gps_radius > 0.0:
        valid_gps_mask = geographic_distances <= gps_radius

    recall = [0] * NUM_NEIGHBORS
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)
    top1_geo_distance = 0.0

    if log:
        log_path = os.path.join('./log', location_name, model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        out_fp_file_name = os.path.join(log_path, f"db{m}_q{n}_log_{embedding_type}_fp.txt")
        with open(out_fp_file_name, 'w') as f:
            s = 'query_idx,query,fp,fp_emb_dist,fp_world_dist,tp,tp_emb_dist,tp_world_dist\n'
            f.write(s)
        out_top5_file_name = os.path.join(log_path, f"db{m}_q{n}_log_{embedding_type}_search_results.txt")
        with open(out_top5_file_name, 'w') as f:
            s = 'query_idx,query,northing,easting,top1,top1_emb_dist,top1_geo_dist,top1_is_match,'
            s += 'top2,top2_emb_dist,top2_geo_dist,top2_is_match,top3,top3_emb_dist,top3_geo_dist,top3_is_match,'
            s += 'top4,top4_emb_dist,top4_geo_dist,top4_is_match,top5,top5_emb_dist,top5_geo_dist,top5_is_match\n'
            f.write(s)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = faiss_distances[i], faiss_predictions[i]
        # Filter only candidates within valid GPS range (slower so only run if needed)
        if valid_gps_mask is not None:
            valid_indices = np.nonzero(valid_gps_mask[i])[0]
            if len(valid_indices) < NUM_NEIGHBORS:
                raise RuntimeError(f'Found less than {NUM_NEIGHBORS} valid candidates. GPS radius may be too low.')
            valid_distances = embedding_distances[i, valid_indices]
            sort_indices = np.argsort(valid_distances)
            indices = valid_indices[sort_indices][:NUM_NEIGHBORS]
            distances = embedding_distances[i, indices]

        if log:
            file_mode = 'a'
            query_key = 'image' if embedding_type == 'image_embeddings' else 'query'
            # Log false positives (returned as the first element)
            # Check if there's a false positive returned as the first element
            if indices[0] not in true_neighbors:
                fp_ndx = indices[0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0]  # Distance in embedding space
                fp_world_dist = geographic_distances[i, indices[0]]
                # Find the first true positive
                tp = None
                for k in range(len(indices)):
                    if indices[k] in true_neighbors:
                        closest_pos_ndx = indices[k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[k]
                        tp_world_dist = geographic_distances[i, indices[k]]
                        break
                            
                out_fp_file_name = os.path.join(log_path, f"db{m}_q{n}_log_{embedding_type}_fp.txt")
                with open(out_fp_file_name, file_mode) as f:
                    s = '{},{},{},{:0.2f},{:0.2f}'.format(i, query_details[query_key], fp[query_key], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ',0,0,0\n'
                    else:
                        s += ',{},{:0.2f},{:0.2f}\n'.format(tp[query_key], tp_emb_dist, tp_world_dist)
                    f.write(s)

            # Save details of 5 best matches for later visualization for 1% of queries
            s = f"{i},{query_details[query_key]},{query_details['northing']},{query_details['easting']}"
            for k in range(min(len(indices), 5)):
                is_match = indices[k] in true_neighbors
                e_ndx = indices[k]
                e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                e_emb_dist = distances[k]
                world_dist = geographic_distances[i, indices[k]]
                s += f",{e[query_key]},{e_emb_dist:0.2f},{world_dist:0.2f},{1 if is_match else 0}"
            s += '\n'
            out_top5_file_name = os.path.join(log_path, f"db{m}_q{n}_log_{embedding_type}_search_results.txt")
            with open(out_top5_file_name, file_mode) as f:
                f.write(s)

        for j in range(len(indices)):
            if indices[j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j+1)
                break

        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

        top1_geo_distance += geographic_distances[i, indices[0]]

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1/np.array(recall_idx))*100
    mean_top1_geo_distance = (top1_geo_distance/float(num_evaluated))
    return recall, one_percent_recall, mrr, mean_top1_geo_distance


def print_eval_stats(stats):
    # Altered to expect per-split stats
    msg = 'Eval Results'
    for database_name in stats:
        msg += '\nDataset: {}'.format(database_name)
        for split in stats[database_name]:
            msg += '\n    Split: {}'.format(split)
            t = '\n    MULTI-MODAL -- Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. top1 geo dist: {:.2f}m   Avg. recall @N:\n    '
            msg += t.format(
                stats[database_name][split]['ave_one_percent_recall'],
                stats[database_name][split]['ave_mrr'],
                stats[database_name][split]['ave_top1_geo_dist'],
            )
            for x in stats[database_name][split]['ave_recall']:
                msg += f"{x:.2f},"
            t = '\n    IMAGE -- Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. top1 geo dist: {:.2f}m   Avg. recall @N:\n    '
            msg += t.format(
                stats[database_name][split]['ave_one_percent_recall_image'],
                stats[database_name][split]['ave_mrr_image'],
                stats[database_name][split]['ave_top1_geo_dist_image'],
            )
            for x in stats[database_name][split]['ave_recall_image']:
                msg += f"{x:.2f},"
            t = '\n    CLOUD -- Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. top1 geo dist: {:.2f}m   Avg. recall @N:\n    '
            msg += t.format(
                stats[database_name][split]['ave_one_percent_recall_cloud'],
                stats[database_name][split]['ave_mrr_cloud'],
                stats[database_name][split]['ave_top1_geo_dist_cloud'],
            )
            for x in stats[database_name][split]['ave_recall_cloud']:
                msg += f"{x:.2f},"
            msg += '\n'
    logging.info(msg)


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            s += f"\n[{ds}]"
            for split in stats[ds]:
                s += f'\n    Split: [{split}]'
                ave_1p_recall = stats[ds][split]['ave_one_percent_recall']
                ave_recall = stats[ds][split]['ave_recall'][0]
                ave_mrr = stats[ds][split]['ave_mrr']
                ave_top1_geo_dist = stats[ds][split]['ave_top1_geo_dist']
                ave_1p_recall_image = stats[ds][split]['ave_one_percent_recall_image']
                ave_recall_image = stats[ds][split]['ave_recall_image'][0]
                ave_mrr_image = stats[ds][split]['ave_mrr_image']
                ave_top1_geo_dist_image = stats[ds][split]['ave_top1_geo_dist_image']
                ave_1p_recall_cloud = stats[ds][split]['ave_one_percent_recall_cloud']
                ave_recall_cloud = stats[ds][split]['ave_recall_cloud'][0]
                ave_mrr_cloud = stats[ds][split]['ave_mrr_cloud']
                ave_top1_geo_dist_cloud = stats[ds][split]['ave_top1_geo_dist_cloud']
                s += '\n    MULTI-MODAL -- AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, Top-1 Geo Dist: {:0.2f}m, AR@N:\n    '.format(
                    ave_1p_recall, ave_recall, ave_mrr, ave_top1_geo_dist
                )
                for x in stats[ds][split]['ave_recall']:
                    s += f"{x:0.2f},"
                s += '\n    IMAGE -- AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, Top-1 Geo Dist: {:0.2f}m, AR@N:\n    '.format(
                    ave_1p_recall_image, ave_recall_image, ave_mrr_image, ave_top1_geo_dist_image
                )
                for x in stats[ds][split]['ave_recall_image']:
                    s += f"{x:0.2f},"
                s += '\n    CLOUD -- AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, Top-1 Geo Dist: {:0.2f}m, AR@N:\n    '.format(
                    ave_1p_recall_cloud, ave_recall_cloud, ave_mrr_cloud, ave_top1_geo_dist_cloud
                )
                for x in stats[ds][split]['ave_recall_cloud']:
                    s += f"{x:0.2f},"
                s += '\n'
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true', help="Log false positives and top-5 retrievals")
    parser.set_defaults(log=False)
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))
    parser.add_argument('--gps_radius', type=float, default=0.0,
                        help='Pseudo-GPS radius in metres to limit database search area. Defaults to 0.0 (entire database)')
    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    if args.save_embeddings and args.load_embeddings:
        print('[WARNING] Both save_embeddings AND load_embeddings specified, which is redundant. '
              'Will proceed without saving embeddings.')
        args.save_embeddings = False
    print(f'Save embeddings: {args.save_embeddings}')
    print(f'Load embeddings: {args.load_embeddings}')
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    if args.gps_radius < 0.0:
        raise ValueError('Invalid GPS radius (must be positive)')
    print('Pseudo GPS Radius: {}'.format(args.gps_radius))
    print('')

    set_seed()  # Seed RNG

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        model_name = os.path.splitext(os.path.split(args.weights)[1])[0]
        print('Loading weights: {}'.format(args.weights))
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        model_name = w

    model.to(device)

    logging_level = 'DEBUG' if params.debug else 'INFO'
    create_logger(log_file=None, logging_level=logging_level)

    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    prefix = "Model Params: {}, Config: {}, Model: {}".format(model_params_name, config_name, model_name)

    stats = evaluate(model, device, params, args.log, model_name, show_progress=True,
                     save_embeddings=args.save_embeddings,
                     load_embeddings=args.load_embeddings)
    print_eval_stats(stats)

    if args.gps_radius > 0.0:
        pnv_write_eval_stats(f"pnv_{params.dataset_name}_split_multi-modal_gps{args.gps_radius:.1f}m_results.txt", prefix, stats)
    else:
        pnv_write_eval_stats(f"pnv_{params.dataset_name}_split_multi-modal_results.txt", prefix, stats)
