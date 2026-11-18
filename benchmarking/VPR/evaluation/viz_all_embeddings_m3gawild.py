"""
Create visualisations of embeddings with TSNE and heatmaps.
"""

import torch 
import pickle 
import numpy as np 
from tqdm import tqdm 
from evaluation.make_eval_sets_m3gawild import make_eval_sets 
from evaluation.dataloader import get_descriptors_positions, compute_cluster_centres

import faiss 
import pandas as pd 
import argparse 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib import rcParams
import umap

import os
from pathlib import Path
from torchpack.utils.config import configs as CFG 
from models.model_factory import model_factory

VLAD_MAP_SEQUENCES = [  # Using both aerial db and validation ground seqs
    '2024_07_09_aerial_sample10m',
    '2024_07_10_aerial_sample10m',
    '2021_06_22_K-01_ground_sample0.5s',
    '2021_06_21_K-02_ground_sample0.5s',
]

TARGET_SEQUENCES = [
    # 'K-02',
    # 'K-03',
    # 'V-02',
    'V-03',
]
TARGET_QUERIES = [
    # '1623372177.5396628', # v02
    # '1639697497.2286490', # v03
    # '1639700568.95', # v03
    # # '1639700578', # v03
    # # '1639700588', # v03
    # '1624320953.5968955', # k02
    # '1639435392.17', # k03 (open sky)
    # '1639435419.37', # k03 (open sky)

    # Only one we care about:
    '1639700057.3581789', # v03
    # Loose search radius in case exact one is not present:
    # '1639700056', # v03
    '1639700057.3', # v03
    # '1639700058', # v03
]

SAVE_DIR = Path('./viz')

class Evaluator:
    def __init__(self, model, env_data, debug=False):
        self.pos_thresh = CFG.eval_pos_radius
        self.recall_values = list(range(1,26))
        self.env_data = env_data
        self.model = model
        self.image_size = CFG.val_img_size
        self.centre_crop = CFG.centre_crop
        self.debug = debug
        self.batch_size = CFG.batch_size
        
    def get_positives(self, q_coords, db_coords):
        ''' 
        Given the positions of each image in the query and database sets, return 
        the ids of the positive matches for each query 
        '''
        cross_dist = torch.cdist(q_coords, db_coords)
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten().numpy()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return valid_queries, valid_positives
    
    def get_recalls(self, query_data, db_data):
        q_name = query_data['name']
        q_feats = query_data['feats']
        q_coords = query_data['coords']
        q_filenames = query_data['filenames']
        db_name = db_data['name']
        db_feats = db_data['feats']
        db_coords = db_data['coords']
        
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        db_faiss_index.add(db_feats)
        
        q_valid_idx, q_positive_idx = self.get_positives(q_coords, db_coords)
        print(f"{q_name}->{db_name}: {len(q_coords) - len(q_valid_idx)} queries removed due to no valid positives")
        
        # Remove queries, coords for examples with no valid positives before getting predictions
        q_feats = torch.index_select(q_feats, 0, torch.tensor(q_valid_idx))
        q_coords = torch.index_select(q_coords, 0, torch.tensor(q_valid_idx))
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))

        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc=f"{q_name}->{db_name}")):
            q_positives = q_positive_idx[q_idx]
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], q_positives)):
                    recalls[i:] += 1 
                    break 
                    
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100

        if not any([target_sequence in q_name for target_sequence in TARGET_SEQUENCES]):
            return recalls

        save_dir = SAVE_DIR / q_name
        os.makedirs(save_dir, exist_ok=True)

        q_feats = q_feats.numpy()
        q_coords = q_coords.numpy()
        db_feats = db_feats.numpy()
        db_coords = db_coords.numpy()
        # Start positions from origin
        min_coord = np.concatenate([db_coords, q_coords], axis=0).min(axis=0)
        q_coords -= min_coord
        db_coords -= min_coord
        # Store global embedding dist
        print('Computing global embed distances...')
        global_distances = cdist(q_feats, db_feats, metric='euclidean')
        
        # Compute dim reduction on embedding space
        print(f'Computing {args.mode}...', end='', flush=True)
        if args.mode == 'tsne':
            proj_func = TSNE(n_components=2, perplexity=30, random_state=0, metric='euclidean')
        elif args.mode == 'umap':
            proj_func = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, metric='euclidean')
        else:
            raise NotImplementedError
        combined_output = np.concatenate([q_feats, db_feats], axis=0)
        if args.use_pca:
            # First PCA project embeddings to reduce noise in t-SNE/UMAP projection
            pca_min_explained_variance = 0.80
            pca = PCA(n_components=pca_min_explained_variance)
            combined_output = pca.fit_transform(combined_output)
            print(f'{combined_output.shape[-1]} PCA components out of {q_feats.shape[-1]}')
        embeddings_proj = proj_func.fit_transform(combined_output)
        print(' Done!')
        queries_proj, database_proj = np.split(embeddings_proj, [len(q_feats)])

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
        fig.savefig(os.path.join(save_dir, f'_global_{args.mode}.png'))
        fig.savefig(os.path.join(save_dir, f'_global_{args.mode}.svg'))
        fig.savefig(os.path.join(save_dir, f'_global_{args.mode}.pdf'))
        plt.close()

        # Visualise embedding space for query and positives
        for query_idx, query_file in enumerate(q_filenames):
            query_positives = q_positive_idx[query_idx]
            # if i % args.vis_freq == 0 or (false_positive and i-last_false_positive_idx >= args.false_positive_min_freq):
            if any([target_query in query_file for target_query in TARGET_QUERIES]):
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
                # non_retrieval_ndx = np.setdiff1d(np.arange(len(db_coords)), indices[0])
                
                # Add plot of top-k retrieval poses against all poses
                # axs[0].scatter(*np.fliplr(q_coords).T, s=marker_size*0.8, c='grey',
                            #    marker='^', alpha=0.5, label='Queries')  # queries
                axs[0].scatter(*db_coords.T, s=marker_size*0.8,
                            c=global_distances[query_idx], cmap=cmap_proj, marker='o', label='Database')  # db
                # axs[0].scatter(*db_coords[non_retrieval_ndx].T, s=marker_size*0.8,
                #                c=global_distances[query_idx][non_retrieval_ndx], cmap=cmap_proj, marker='o', label='Database')  # db
                axs[0].scatter(*db_coords[np.flipud(predictions[query_idx, :max_cand])].T, s=marker_size*0.4,
                            c=global_distances[query_idx][np.flipud(predictions[query_idx, :max_cand])], vmin=global_distances[query_idx].min(),
                            vmax=global_distances[query_idx].max(), cmap=cmap_proj, edgecolors='#FF0000',
                            marker='D', label=f'Top-{max_cand} Cand')  # top-k retrievals
                axs[0].scatter(*q_coords[query_idx].T, s=marker_size*3, c='#FFFFFF', marker='*',
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
                fig.savefig(os.path.join(save_dir, f'query_{Path(query_file).stem}_{args.mode}.png'))
                fig.savefig(os.path.join(save_dir, f'query_{Path(query_file).stem}_{args.mode}.svg'))
                fig.savefig(os.path.join(save_dir, f'query_{Path(query_file).stem}_{args.mode}.pdf'))
                plt.close()
        
        return recalls

    def run(self) -> pd.DataFrame:
        all_sequences_data = get_descriptors_positions(self.env_data, self.image_size, self.model, self.debug, self.batch_size, centre_crop=self.centre_crop)
        db_data = all_sequences_data.pop(-1)  # NOTE: Assumes 1 DB, and always in last position
        df_results = pd.DataFrame(columns = [f"R@{n}" for n in self.recall_values])
        
        for query_data in all_sequences_data:
            recalls = self.get_recalls(query_data, db_data)
            df_results.loc[query_data['name']] = recalls
            
        df_results = df_results.round(2)
        print(df_results)
        return df_results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_ckpt', type=str, required=False)
    parser.add_argument('--environments', type=str, nargs='+', default=['venman', 'karawatha'], choices=['venman','karawatha'])
    parser.add_argument('--mode', type=str, choices=['tsne', 'umap'], default='tsne', help='Type of dimension reduction for visualising embeddings')
    parser.add_argument('--use_pca', action='store_true', help='Use PCA projection before applying UMAP/tSNE')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args, opts = parser.parse_known_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    CFG.load(args.config)
    CFG.update(opts)
    print(CFG)
    
    model = model_factory(args).cuda()
    model.eval()
    
    venman_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', f'm3gawild_ground-ds-{CFG.anchor_framerate_downsample}_aerial-ds-{CFG.aerial_spatial_downsample}_venman_eval_info.pickle')
    karawatha_eval_info_filepath = os.path.join(os.path.dirname(__file__), 'pickles', f'm3gawild_ground-ds-{CFG.anchor_framerate_downsample}_aerial-ds-{CFG.aerial_spatial_downsample}_karawatha_eval_info.pickle')
    
    if not os.path.exists(venman_eval_info_filepath) or not os.path.exists(karawatha_eval_info_filepath):
        make_eval_sets()

    if CFG.model_name == 'AnyLoc' and CFG.domain == 'map':
        kara_env_data = pickle.load(open(karawatha_eval_info_filepath, 'rb'))
        venman_env_data = pickle.load(open(venman_eval_info_filepath, 'rb'))
        # Pick desired sequences to learn cluster centres from
        vlad_env_data = []
        for seq_data in [*kara_env_data, *venman_env_data]:
            if seq_data['name'] in VLAD_MAP_SEQUENCES:
                vlad_env_data.append(seq_data)
        assert len(vlad_env_data) > 0, 'Invalid sequences'
        compute_cluster_centres(vlad_env_data, CFG.val_img_size, model, args.debug, CFG.batch_size, centre_crop=CFG.centre_crop)
    
    for env in args.environments:
        if env == 'venman':
            env_data = pickle.load(open(venman_eval_info_filepath, 'rb'))
        elif env == 'karawatha':
            env_data = pickle.load(open(karawatha_eval_info_filepath, 'rb'))
    
        df_results = Evaluator(model, env_data, args.debug).run()

        if args.save_dir is not None:
            if args.pretrained_ckpt is not None:
                ckpt_name = Path(args.pretrained_ckpt).stem
            elif CFG.model_name == 'AnyLoc':
                ckpt_name = f'AnyLoc_{CFG.domain}'
            else:
                ckpt_name = 'default'
            save_path = os.path.join(args.save_dir, f"m3gawild_results_{ckpt_name}_{env}.csv")
            df_results.to_csv(save_path)
