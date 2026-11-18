import torch 
import pickle 
import numpy as np 
from tqdm import tqdm 
from evaluation.make_eval_sets_m3gawild import make_eval_sets 
from evaluation.dataloader import get_descriptors_positions, compute_cluster_centres

import faiss 
import pandas as pd 
import argparse 

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
