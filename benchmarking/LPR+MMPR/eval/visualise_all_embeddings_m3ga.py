"""
Script to visualise embedding space, and check clustering of true positive
descriptors. For figures used in M3GA paper.

Ethan Griffiths (Data61, Pullenvale)
"""
import pickle
import os
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch
import tqdm
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib import rcParams
import umap

from dataset.base_datasets import get_pointcloud_loader
from models.model_factory import model_factory
from misc.logger import create_logger
from misc.utils import TrainingParams
from misc.torch_utils import set_seed
from misc.point_clouds import plot_points
from eval.utils import get_query_database_splits
from eval.pnv_evaluate import (
    get_latent_vectors,
    save_embeddings_to_file,
    load_embeddings_from_file,
)

TARGET_SEQUENCES = [
    # 'K-02',
    # 'K-03',
    'V-02',
    'V-03',
]
TARGET_QUERIES = [
    '1623372177.5396628', # v02
    '1639697497.2286490', # v03
    '1639700057.3581789', # v03
    '1639700568.95', # v03
    # '1639700578', # v03
    # '1639700588', # v03
    '1624320953.5968955', # k02
    '1639435392.17', # k03 (open sky)
    '1639435419.37', # k03 (open sky)
]

def evaluate(model, device, params: TrainingParams, save_dir: str, log: bool = False,
             model_name: str = 'model', show_progress: bool = False, 
             save_embeddings: bool = False, load_embeddings: bool = False):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)
    stats = {}
    ave_recall = []
    ave_one_percent_recall = []
    ave_mrr = []
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
                                save_dir=save_dir, log=log, model_name=model_name,
                                show_progress=show_progress, save_embeddings=save_embeddings,
                                load_embeddings=load_embeddings)
        stats[location_name] = temp
        ave_one_percent_recall.append(temp['average']['ave_one_percent_recall'])
        ave_recall.append(temp['average']['ave_recall'])
        ave_mrr.append(temp['average']['ave_mrr'])
        
    # Compute average stats
    stats['average'] = {'average': {'ave_one_percent_recall': np.mean(ave_one_percent_recall),
                                    'ave_recall': np.mean(ave_recall, axis=0),
                                    'ave_mrr': np.mean(ave_mrr)}}
    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets,
                     location_name: str, save_dir: str, log: bool = False,
                     model_name: str = 'model', show_progress: bool = False,
                     save_embeddings: bool = False, load_embeddings: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    stats = {}
    count = 0
    one_percent_recall = []
    mrr = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    save_dir = os.path.join(save_dir, params.dataset_name, location_name, model_name)

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} database embeddings')
    for ii, data_set in enumerate(database_sets):
        temp_embeddings = None
        if len(data_set) > 0:
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'database_{ii}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'database_{ii}')
        database_embeddings.append(temp_embeddings)

    if show_progress:
        logging.info(f'{"Loading" if load_embeddings else "Computing"} query embeddings')
    for jj, data_set in enumerate(query_sets):
        temp_embeddings = None
        if len(data_set) > 0:
            if load_embeddings:
                temp_embeddings = load_embeddings_from_file(model_name, params.dataset_name, location_name, f'query_{jj}')
            else:
                temp_embeddings = get_latent_vectors(model, data_set, device, params, show_progress)
            if save_embeddings:
                save_embeddings_to_file(temp_embeddings, model_name, params.dataset_name, location_name, f'query_{jj}')
        query_embeddings.append(temp_embeddings)

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
            if not any([target_sequence in split_name for target_sequence in TARGET_SEQUENCES]):
                continue
            pair_recall, pair_opr, pair_mrr = get_recall(i, j, database_embeddings,
                                                         query_embeddings, query_sets,
                                                         database_sets,
                                                         params=params,
                                                         save_dir=save_dir,
                                                         log=log, model_name=model_name)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)
            
            # Report per-split metrics
            stats[split_name] = {'ave_one_percent_recall': pair_opr,
                                 'ave_recall': pair_recall,
                                 'ave_mrr': pair_mrr}
    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    ave_mrr = np.mean(mrr)
    stats['average'] = {'ave_one_percent_recall': ave_one_percent_recall,
                        'ave_recall': ave_recall,
                        'ave_mrr': ave_mrr}
    return stats


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets,
               params: TrainingParams, save_dir: str, log=False, model_name: str = 'model'):
    # Make save dir
    save_dir = os.path.join(save_dir, f'db{m}_q{n}')
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(database_vectors[m], dict):
        database_output = database_vectors[m][args.embedding_type]
        queries_output = query_vectors[n][args.embedding_type]
    else:
        database_output = database_vectors[m]
        queries_output = query_vectors[n]

    # Store global embedding dist
    global_distances = cdist(queries_output, database_output, metric='euclidean')

    # Create array of coordinates of all query and db elements
    database_positions = np.array([(db_details['northing'], db_details['easting']) for db_details in database_sets[m].values()])
    query_positions = np.array([(query_details['northing'], query_details['easting']) for query_details in query_sets[n].values()])
    geographic_distances = cdist(query_positions, database_positions, metric='euclidean')

    # Start positions from origin
    min_coord = np.concatenate([database_positions, query_positions], axis=0).min(axis=0)
    database_positions -= min_coord
    query_positions -= min_coord

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    # Compute dim reduction on embedding space
    print(f'Computing {args.mode}...', end='', flush=True)
    if args.mode == 'tsne':
        proj_func = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0, metric='euclidean')
    elif args.mode == 'umap':
        proj_func = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, metric='euclidean')
    else:
        raise NotImplementedError
    combined_output = np.concatenate([queries_output, database_output], axis=0)
    if args.use_pca:
        # First PCA project embeddings to reduce noise in t-SNE/UMAP projection
        pca_min_explained_variance = 0.80
        pca = PCA(n_components=pca_min_explained_variance)
        combined_output = pca.fit_transform(combined_output)
    embeddings_proj = proj_func.fit_transform(combined_output)
    print(' Done!')
    queries_proj, database_proj = np.split(embeddings_proj, [len(queries_output)])

    # Plot tSNE/UMAP for all embeddings
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    marker_size = 20
    legend_fontsize = 16
    # Colourise embeddings by distance in embedding space
    embedding_ds_rate = 2
    queries_proj_ds = queries_proj[::embedding_ds_rate]
    ax.scatter(*queries_proj_ds.T, s=marker_size*0.8, c='red', marker='^', alpha=0.5, label='Queries')  # queries
    database_proj_ds = database_proj[::embedding_ds_rate]
    db_points = ax.scatter(*database_proj_ds.T, s=marker_size*0.8, c='blue', marker='o', alpha=0.5, label='Database')  # db
    ax.legend(prop={'size': legend_fontsize, 'weight': 'bold'}, loc='upper right', markerscale=2)
    # ax.set_title(f'{str.upper(args.mode)} projection on {params.dataset_name} (db{m}-q{n})')
    ax.set_axis_off()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # Save fig
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'_global_{args.mode}_{args.embedding_type}.png'))
    fig.savefig(os.path.join(save_dir, f'_global_{args.mode}_{args.embedding_type}.pdf'))
    plt.close()

    pc_loader = get_pointcloud_loader(params.dataset_name)

    last_false_positive_idx = -1e10
    num_evaluated = 0
    for i in tqdm.tqdm(range(len(queries_output))):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        if m in query_details:  # old tuples store true neighbours directly
            true_neighbors = query_details[m]
        else:  # manually find true neighbors for specified eval radius
            true_neighbors = np.nonzero(geographic_distances[i] <= args.radius)[0].tolist()
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j+1)
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

        # Visualise embedding space for query and positives
        false_positive = indices[0][0] not in true_neighbors
        retrieval_type = 'FP' if false_positive else 'TP'
        # if i % args.vis_freq == 0 or (false_positive and i-last_false_positive_idx >= args.false_positive_min_freq):
        if any([target_query in query_details['query'] for target_query in TARGET_QUERIES]):
            if false_positive:
                last_false_positive_idx = i
            # fig = plt.figure(figsize=(11,8))
            fig = plt.figure(figsize=(5.6,4))
            axs = []
            axs.append(fig.add_subplot(1, 1, 1))
            # axs.append(fig.add_subplot(2, 2, 1))
            # axs.append(fig.add_subplot(2, 2, 2))
            # axs.append(fig.add_subplot(2, 2, 3, projection='3d'))
            # axs.append(fig.add_subplot(2, 2, 4, projection='3d'))
            marker_size = 28
            cmap_proj = 'viridis_r'
            cmap_points = 'jet'
            point_size = 4
            legend_fontsize = 14
            max_cand = 5
            # # Get indices of non-topk retrievals
            # non_retrieval_ndx = np.setdiff1d(np.arange(len(database_positions)), indices[0])
            
            # Add plot of top-k retrieval poses against all poses
            # axs[0].scatter(*np.fliplr(query_positions).T, s=marker_size*0.8, c='grey',
                        #    marker='^', alpha=0.5, label='Queries')  # queries
            axs[0].scatter(*np.fliplr(database_positions).T, s=marker_size*0.8,
                           c=global_distances[i], cmap=cmap_proj, marker='o', label='Database')  # db
            # axs[0].scatter(*np.fliplr(database_positions[non_retrieval_ndx]).T, s=marker_size*0.8,
            #                c=global_distances[i][non_retrieval_ndx], cmap=cmap_proj, marker='o', label='Database')  # db
            axs[0].scatter(*np.fliplr(database_positions[np.flipud(indices[0][:max_cand])]).T, s=marker_size*0.4,
                           c=global_distances[i][np.flipud(indices[0][:max_cand])], vmin=global_distances[i].min(),
                           vmax=global_distances[i].max(), cmap=cmap_proj, edgecolors='#FF0000',
                           marker='D', label=f'Top-{max_cand} Cand')  # top-k retrievals
            axs[0].scatter(*np.flip(query_positions[i], axis=0).T, s=marker_size*3, c='#FFFFFF', marker='*',
                           edgecolors='black', label='Anchor')  # anchor
            axs[0].legend(prop={'size': legend_fontsize, 'weight': 'bold'}, markerscale=2)
            # axs[0].set_title(f'Retrievals on {params.dataset_name} (db{m}-q{n})')
            axs[0].set_xlabel('x [m]')
            axs[0].set_ylabel('y [m]')
            axs[0].set_aspect('equal', adjustable='box')
            
            # cbar = fig.colorbar(db_points, ax=axs[0])  # Add colorbar to embedding plot (colormap is the same)
            # cbar.ax.set_ylabel('Embedding L2 Distance', rotation=-90, va="bottom")

            # Save fig
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'query_{Path(query_details["query"]).stem}_{retrieval_type}_{args.mode}_{args.embedding_type}.png'))
            fig.savefig(os.path.join(save_dir, f'query_{Path(query_details["query"]).stem}_{retrieval_type}_{args.mode}_{args.embedding_type}.pdf'))
            plt.close()

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1/np.array(recall_idx))*100
    return recall, one_percent_recall, mrr


def print_eval_stats(stats):
    # Altered to expect per-split stats
    msg = 'Eval Results'
    for database_name in stats:
        msg += '\nDataset: {}\n'.format(database_name)
        for split in stats[database_name]:
            msg += '    Split: {}\n'.format(split)
            t = '    Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. recall @N:\n'
            msg += t.format(
                stats[database_name][split]["ave_one_percent_recall"],
                stats[database_name][split]["ave_mrr"],
            )
            msg += '    ' + str(stats[database_name][split]['ave_recall']).replace('\n','\n    ')
    logging.info(msg)


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    # ave_1p_recall_l = []
    # ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            s += f"\n[{ds}]\n"
            for split in stats[ds]:
                s += f'    Split: [{split}]\n'
                ave_1p_recall = stats[ds][split]['ave_one_percent_recall']
                # ave_1p_recall_l.append(ave_1p_recall)
                ave_recall = stats[ds][split]['ave_recall'][0]
                # ave_recall_l.append(ave_recall)
                ave_mrr = stats[ds][split]['ave_mrr']
                s += '    AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, AR@N:\n'.format(ave_1p_recall, ave_recall, ave_mrr)
                s += '    ' + str(stats[ds][split]['ave_recall']).replace('\n','\n    ')
                s += '\n'

        # NOTE: below is redundant as average is now stored in stats dict
        # mean_1p_recall = np.mean(ave_1p_recall_l)
        # mean_recall = np.mean(ave_recall_l)
        # s += "\n\nMean AR@1%: {:0.2f}, Mean AR@1: {:0.2f}\n\n".format(mean_1p_recall, mean_recall)
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--mode', type=str, choices=['tsne', 'umap'], default='tsne', help='Type of dimension reduction for visualising embeddings')
    parser.add_argument('--use_pca', action='store_true', help='Use PCA projection before applying UMAP/tSNE')
    parser.add_argument('--embedding_type', type=str, choices=['embeddings', 'image_embeddings', 'cloud_embeddings'], default='embeddings',
                        help='Which embedding type to use (for multi-modal models)')
    parser.add_argument('--vis_freq', type=int, default=50, help='Frequency to visualise embeddings for a given query (i.e. visualise every nth query)')
    parser.add_argument('--false_positive_min_freq', type=int, default=20, help='Min frequency between saving false positive visualisations')
    parser.add_argument('--radius', type=float, required=True, help='True positive threshold in metres')
    parser.add_argument('--save_dir', type=str, default='./vis_embeddings_m3ga', help='Path to save embedding visualisations')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--log', dest='log', action='store_true', help="Log false positives and top-5 retrievals")
    parser.set_defaults(log=False)
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings to disk')
    parser.add_argument('--load_embeddings', action='store_true',
                        help=('Load embeddings from disk. Note this script will only check if '
                              'weights paths match, not if the configs used match.'))

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Mode: {}'.format(args.mode))
    print('Use PCA: {}'.format(args.use_pca))
    print('Vis freq: {}'.format(args.vis_freq))
    print('False positive min freq: {}'.format(args.false_positive_min_freq))
    print('Radius: {}'.format(args.radius))
    print('Save dir: {}'.format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_embeddings and args.load_embeddings:
        print('[WARNING] Both save_embeddings AND load_embeddings specified, which is redundant. '
              'Will proceed without saving embeddings.')
        args.save_embeddings = False
    print(f'Save embeddings: {args.save_embeddings}')
    print(f'Load embeddings: {args.load_embeddings}')
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
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

    stats = evaluate(model, device, params, args.save_dir, args.log, model_name, show_progress=True,
                     save_embeddings=args.save_embeddings, load_embeddings=args.load_embeddings)
    print_eval_stats(stats)

    pnv_write_eval_stats(f"pnv_{params.dataset_name}_split_results.txt", prefix, stats)


