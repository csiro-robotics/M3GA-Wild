import os 
import pandas as pd 
import torch 
import argparse 
import numpy as np 
import importlib 
from evaluate_m3gawild.dataloader import get_image_lidar_features
import faiss 
from tqdm import tqdm 

class Evaluator:
    def __init__(self, model, CFG, environment, debug):
        self.model = model 
        self.CFG = CFG 
        self.environment = environment 
        self.debug = debug 
        self.recall_values = list(range(1,26))
        self.pos_thresh = 30.0
        
    def get_positives(self, q_coords, db_coords):
        cross_dist = torch.cdist(q_coords, db_coords)
        valid_queries = []
        valid_positives = []
        for idx, row in enumerate(cross_dist):
            positives = torch.nonzero(row <= self.pos_thresh).flatten().numpy()
            if len(positives) == 0:
                continue 
            valid_queries.append(idx)
            valid_positives.append(positives)
            
        return torch.tensor(valid_queries, dtype=torch.int32), valid_positives
    
    def get_recalls(self, query_info, db_info):
        q_name = query_info['name']
        q_feats = torch.tensor(query_info['feats'])
        q_coords = torch.tensor(query_info['coords'])
        db_name = db_info['name']
        db_feats = torch.tensor(db_info['feats'])
        db_coords =torch.tensor(db_info['coords'])
        
        db_faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
        db_faiss_index.add(db_feats)
        
        valid_split_idx, positive_idx = self.get_positives(q_coords, db_coords)
        
        print(f"{q_name}->{db_name}: {len(q_coords) - len(valid_split_idx)} removed due to no valid positives")
        
        q_feats = torch.index_select(q_feats, 0, valid_split_idx)
        q_coords = torch.index_select(q_coords, 0, valid_split_idx)
        
        _, predictions = db_faiss_index.search(q_feats, max(self.recall_values))
        
        vis_recall_info = {}
        
        # Get recalls 
        recalls = np.zeros(len(self.recall_values))
        for q_idx, preds in enumerate(tqdm(predictions, desc = f"{q_name}->{db_name}")):
            vis_recall_info[valid_split_idx[q_idx]] = [
                preds[0].item(), preds[0].item() in positive_idx[q_idx]
            ]    
            
            for i, n in enumerate(self.recall_values):
                if np.any(np.isin(preds[:n], positive_idx[q_idx])):
                    recalls[i:] += 1 
                    break 
        
        # Divide by number of queries and multiply by 100 
        recalls = recalls / len(q_feats) * 100
        print(f"{q_name}->{db_name} : {recalls.round(2)}")
        return recalls, vis_recall_info
        
    def run(self):
        query_info_list, db_info = get_image_lidar_features(
            self.model, self.CFG, self.environment, self.debug)

        df_results = pd.DataFrame(columns=[f"R@{n}" for n in self.recall_values])
        vis_recall_results = {}
        for query_info in query_info_list:
            recalls, vis_recall_info = self.get_recalls(query_info, db_info)
            df_results.loc[query_info['name']] = recalls
            vis_recall_results[query_info['name']] = vis_recall_info
        df_results.loc['Average'] = df_results.mean(0)
        df_results = df_results.round(2)
        print(df_results)
        return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--environments', type=str, default=['Venman','Karawatha','QCAT','Samford'], nargs='+')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--eval_mode', type=str, default='test', choices=['test', 'train', 'both'],
                        help='whether to evaluate on train or test set')
    args = parser.parse_args()

    CFG = importlib.import_module(f"config.{args.expid}").CFG
    model = importlib.import_module(f"models.{CFG.model}").Model(CFG)
    if args.ckpt is None:
        print('No checkpoint loaded -- using initial weights')
    else:
        ckpt = torch.load(args.ckpt)
        # ckpt = {k:v for k,v in ckpt.items() if "pooling" not in k}  # not sure why this is here?
        model.load_state_dict(ckpt)
    model = model.to('cuda')
    model.eval()
    print(f"Loaded pretrained model from {args.ckpt}")

    if args.eval_mode == 'both':
        eval_modes = ['train', 'test']
    else:
        eval_modes = [args.eval_mode]

    for eval_mode in eval_modes:
        CFG.eval_mode = eval_mode
        environments = args.environments
        if eval_mode == 'train':  # QCAT and Samford have no training regions
            environments = [env for env in args.environments if env not in ['QCAT', 'Samford']]
        for env in environments:
            evaluator = Evaluator(model, CFG, env, args.debug)
            df_results = evaluator.run()
            
            if args.save_dir is not None:
                if not os.path.exists(args.save_dir):
                    print(f"Save Directory did not exist. Creating {args.save_dir}")
                    os.makedirs(args.save_dir)
                save_path = os.path.join(args.save_dir, f"crossmodal_results_m3ga_{env}_{eval_mode}.csv")
                df_results.to_csv(save_path)
