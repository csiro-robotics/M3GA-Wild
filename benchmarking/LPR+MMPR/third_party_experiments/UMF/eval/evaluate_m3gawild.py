import numpy as np
import pickle
import os
import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
import tqdm
from misc.utils import UMFParams
from models.model_factory import model_factory
from dataset.m3gawild import M3GAWildEvalDataset
from dataset.dataset_utils import make_eval_collate_fn
from dataset.augmentation import ValRGBTransform
import dotsi
import yaml
from eval.utils import load_config, compute_and_log_stats_per_split

LOAD_FEATURES = False


def load_config(config_file):
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def evaluate(model, device, params, model_name: str, silent=True):
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    stats_img = {}
    stats_pc = {}
    stats_combined = {}
    ave_recall = []
    ave_similarity = []
    ave_one_percent_recall = []

    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[1]
        temp = query_file.split('_')[1]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        if not silent:
            print(f'Evaluating: {location_name}')
        temp, temp_img, temp_pc, temp_combined = evaluate_dataset(
            model, device, params, model_name, database_sets, query_sets, location_name=location_name, silent=silent
        )
        stats[location_name] = temp
        ave_recall.append(temp['average']['ave_recall'])
        ave_similarity.append(temp['average']['ave_similarity'])
        ave_one_percent_recall.append(temp['average']['ave_one_percent_recall'])
        stats_img[location_name] = temp_img
        stats_pc[location_name] = temp_pc
        stats_combined[location_name] = temp_combined

    # Compute average stats
    stats['average'] = {'average': {'ave_recall': np.mean(ave_recall, axis=0),
                                    'ave_similarity': np.mean(ave_similarity),
                                    'ave_one_percent_recall': np.mean(ave_one_percent_recall)}}
    return stats, stats_img, stats_pc, stats_combined

def evaluate_dataset(model, device, params, model_name: str, database_sets, query_sets, location_name, silent=True):
    features_folder = Path(__file__).parent / 'features' / model_name / location_name
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

    initialize_model(model, database_sets, features_folder, device, params)

    if not LOAD_FEATURES:
        # Process and save database sets
        for idx, set in enumerate(tqdm.tqdm(database_sets, disable=silent)):
            out = get_latent_vectors(model, set, device, params, dim_reduction=True, normalize=False)
            with open(os.path.join(features_folder, f'db_set_{idx}.pkl'), 'wb') as f:
                pickle.dump(out, f)

        # Process and save query sets
        for idx, set in enumerate(tqdm.tqdm(query_sets, disable=silent)):
            out = get_latent_vectors(model, set, device, params, dim_reduction=True, normalize=False)
            with open(os.path.join(features_folder, f'query_set_{idx}.pkl'), 'wb') as f:
                pickle.dump(out, f)

    stats = [None]*4
    mode = params.model_params.params['mode']

    modalities = ['base']
    if mode != 'fusion':
        modalities.extend(['img', 'pc', 'combined'])

    for ii, modality in enumerate(modalities):        
        stat = compute_and_log_stats_per_split(mode, modality, query_sets, database_sets, features_folder)
        stats[ii] = stat
    
    return tuple(stats)

def get_latent_vectors(model, set, device, params, dim_reduction=True, normalize=False):
    if len(set) == 0:
        return None
    cfg = load_config(params.model_params_path)
    cfg = dotsi.Dict(cfg)

    val_image_transform = ValRGBTransform(image_size=cfg.model.visual.image_size)
    eval_dataset = M3GAWildEvalDataset(set=set, params=params, cfg=cfg, image_transform=val_image_transform)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=params.val_batch_size,
        collate_fn=make_eval_collate_fn(dataset=eval_dataset),
        num_workers=params.num_workers,
        shuffle=False,
    )

    model.eval()
    embeddings_l = []

    feature_lists = {
        'img_fine_feat': [], 'img_attns': [], 'pc_fine_feat': [], 'pc_attns': [],
        'img_super_feat': [], 'img_strengths': [], 'pc_super_feat': [], 'pc_strengths': []
    }

    for batch in eval_dataloader:
        if not dim_reduction and np.random.rand() > 0.1:
            continue

        batch = {e: batch[e].to(device) for e in batch}
        with torch.no_grad():
            outputs = model(batch, dim_reduction=dim_reduction, normalize=normalize)
            if params.normalize_embeddings:
                outputs['embedding'] = torch.nn.functional.normalize(outputs['embedding'], p=2, dim=1)
        embeddings = outputs['embedding'].detach().cpu().numpy()
        embeddings_l.extend(embeddings)

        for key in feature_lists.keys():
            if key in outputs:
                feature_lists[key].extend(outputs[key].cpu())
    torch.cuda.empty_cache()

    out = {'embedding': torch.tensor(np.array(embeddings_l), dtype=torch.float32)}
    for key, value in feature_lists.items():
        if value:
            out[key] = torch.stack(value, dim=0) if 'feat' in key else torch.cat(value, dim=0)

    return out

def print_eval_stats(stats):
    # Altered to expect per-split stats
    msg = 'Eval Results'
    for database_name in stats:
        if stats[database_name] is None:
            continue
        msg += '\nDataset: {}'.format(database_name)
        for split in stats[database_name]:
            msg += '\n    Split: {}\n'.format(split)
            t = '    Avg. top 1% recall: {:.2f}   Avg. similarity: {:.2f}   Recall@N:\n'
            msg += t.format(
                stats[database_name][split]['ave_one_percent_recall'],
                stats[database_name][split]['ave_similarity'],
            )
            recall = stats[database_name][split]['ave_recall']
            # msg += f'    Recall@N:\n'
            msg += f'    '
            for ii, x in enumerate(recall):
                msg += f'{x:0.2f},'
                # if (ii+1) % 5 == 0 and (ii+1) < len(recall):  # no indenting for easier copy/paste
                #     msg += '\n              '
            msg += '\n'
    print(msg, flush=True)

def write_eval_recall(stats, model_name: str, params: UMFParams):
    # Altered to expect per-split stats
    csv_file = Path(__file__).parent / 'results' / f'{model_name}.csv'
    if not csv_file.parent.exists():
        os.makedirs(csv_file.parent)
    msg = f'# Params: {params.params_path}    Model Params: {params.model_params_path}'
    msg += '\n,R@1,R@2,R@3,R@4,R@5,R@6,R@7,R@8,R@9,R@10,R@11,R@12,R@13,R@14,R@15,R@16,R@17,R@18,R@19,R@20,R@21,R@22,R@23,R@24,R@25'
    for database_name in stats:
        if stats[database_name] is None:
            continue
        for split in stats[database_name]:
            recall = stats[database_name][split]['ave_recall']
            recall_rounded = [str(val) for val in recall.round(2)]
            if database_name == 'average' and split == 'average':
                msg += f'\ntotal average,{",".join(recall_rounded)}'
            else:
                msg += f'\n{split},{",".join(recall_rounded)}'
    with open(csv_file, 'w') as f:
        f.write(msg)

def initialize_model(model, train_set, features_folder, device, params):
    model.eval()
    mode = params.model_params.params['mode']
    if mode not in ['ransac', 'superfeatures']:
        return None
    des_img_train = []
    des_pc_train = []
    train_set = train_set[:]

    if not LOAD_FEATURES:
        for set in train_set:
            out = get_latent_vectors(model, set, device, params, dim_reduction=False, normalize=False)

            if out is None:
                continue
            if 'img_fine_feat' in out:
                des_img_train.append(out['img_fine_feat'].detach().cpu())
                des_pc_train.append(out['pc_fine_feat'].detach().cpu())

            elif 'img_super_feat' in out:
                des_img_train.append(out['img_super_feat'].detach().cpu())
                des_pc_train.append(out['pc_super_feat'].detach().cpu())

        des_img_train = torch.cat(des_img_train, dim=0)
        des_pc_train = torch.cat(des_pc_train, dim=0)

        if mode == 'ransac':
            pc_feat_dim = des_pc_train.shape[2]
            img_feat_dim = des_img_train.shape[2]
        elif mode == 'superfeatures':
            pc_feat_dim = des_pc_train.shape[2]
            img_feat_dim = des_img_train.shape[2]
        
        max_num_samples = 50000
        if des_img_train.shape[0] > max_num_samples:
            indices_img = np.random.choice(des_img_train.shape[0], max_num_samples, replace=False)
            des_img_train_sampled = des_img_train[indices_img].reshape(-1, img_feat_dim)
        else:
            des_img_train_sampled = des_img_train.reshape(-1, img_feat_dim)
        
        
        if des_pc_train.shape[0] > max_num_samples:
            indices_pc = np.random.choice(des_pc_train.shape[0], max_num_samples, replace=False)
            des_pc_train_sampled = des_pc_train[indices_pc].reshape(-1, pc_feat_dim)
        else:
            des_pc_train_sampled = des_pc_train.reshape(-1, pc_feat_dim)

        if mode == 'ransac':
            mi, Pi = model.lt.reduction_layer.initialize_pca_whitening(des_img_train_sampled)
            mp, Pp = model.lt3d.reduction_layer.initialize_pca_whitening(des_pc_train_sampled)
        elif mode == 'superfeatures':
            mi, Pi = model.lit.reduction_layer.initialize_pca_whitening(des_img_train_sampled)
            mp, Pp = model.lit3d.reduction_layer.initialize_pca_whitening(des_pc_train_sampled)

        del des_img_train, des_pc_train, des_img_train_sampled, des_pc_train_sampled
        
        with open(f"{features_folder}/initialization.pkl", 'wb') as f:
            pickle.dump({'mi': mi, 'Pi': Pi, 'mp': mp, 'Pp': Pp}, f)
    
    # load initialization values
    with open(f"{features_folder}/initialization.pkl", 'rb') as f:
        init = pickle.load(f)
    mi, Pi, mp, Pp = init['mi'], init['Pi'], init['mp'], init['Pp']

    if hasattr(model, 'lit'):
        print('-> Loading initialization values for lit and lit3d')
        model.lit.reduction_layer.load_initialization(mi, Pi)
        model.lit3d.reduction_layer.load_initialization(mp, Pp)
    elif hasattr(model, 'lt'):
        print('-> Loading initialization values for lt and lt3d')
        model.lt.reduction_layer.load_initialization(mi, Pi)
        model.lt3d.reduction_layer.load_initialization(mp, Pp)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on M3GA-Wild dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Trained model weights')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        sys.exit('Please provide a path to the trained model weights')

    params = UMFParams(args.config, args.model_config)
    params.print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))

        pretrained_state_dict = torch.load(args.weights, map_location='cpu')
        dim_reduction_prefix = '.reduction_layer.'
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if dim_reduction_prefix not in k}
        model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    model_name = Path(args.weights).stem

    stats, stats_img, stats_pc, stats_combined = evaluate(
        model, device, params, model_name=model_name, silent=False
    )
    print("----- Without reranking -----")

    print_eval_stats(stats)
    write_eval_recall(stats, model_name, params)

    print("----- With Img reranking -----")
    print_eval_stats(stats_img)

    print("----- With PC reranking -----")
    print_eval_stats(stats_pc)

    print("----- With Combined reranking -----")
    print_eval_stats(stats_combined)
