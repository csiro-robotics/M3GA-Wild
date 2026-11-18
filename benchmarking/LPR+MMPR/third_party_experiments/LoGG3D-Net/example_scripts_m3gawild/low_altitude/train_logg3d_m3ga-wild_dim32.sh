#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=3
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

node_1=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
echo "node_1="$node_1
node_2=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 2p)
echo "node_2="$node_2
node_3=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 3p)
echo "node_3="$node_3
export _HOSTS="${node_1}:2,${node_2}:2,${node_3}:2"
echo "hosts="$_HOSTS

export _NGPU=6

pixi run torchpack dist-run -np ${_NGPU} -H ${_HOSTS} -v python training/train.py \
    --train_pipeline 'LOGG3D' \
	--dataset 'GeneralPointSparseTupleDataset' \
	--gp_rem False \
	--cs_wildplaces_dir ../../../../data/M3GA-Wild \
	--train_pickles "../../../../data/M3GA-Wild/training_queries_M3GAWild-aerial_baseline_v4.pickle" \
	--train_dataset_name "m3gawild" \
	--train_num_workers "6" \
	--negatives_per_query "9" \
	--use_random_rotation "True" \
	--feature_dim "32" \
	--voxel_size 0.50 \
	--max_epoch "50"
