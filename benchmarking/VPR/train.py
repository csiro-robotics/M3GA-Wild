import os
import argparse
import torch
from datetime import datetime
import wandb
import random
from pathlib import Path
import numpy as np
from torchpack.utils.config import configs as CFG

from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler

from data.datamodule import WildCrossDataModule, M3GAWildDataModule
from models.model_factory import model_factory


def train(args):
    seed_everything(CFG.seed)
    
    model = model_factory(args)
    if CFG.dataset.lower() == 'wildcross':
        datamodule = WildCrossDataModule()
    elif CFG.dataset.lower() == 'm3ga-wild':
        datamodule = M3GAWildDataModule()
    else:
        raise NotImplementedError(f'{CFG.dataset} dataset not implemented')
    
    wandb_logger = WandbLogger(save_dir=args.save_dir, project='WildCross VPR')
    wandb_logger.log_hyperparams(CFG.dict())
    
    # Define the checkpointing callback
    curr_time = datetime.fromtimestamp(wandb_logger.experiment._start_time).strftime("%Y%m%d_%H%M%S")
    exp_name = f'{curr_time}-{wandb_logger.version}'
    checkpointing = callbacks.ModelCheckpoint(
        dirpath=Path(args.save_dir) / 'checkpoints' / exp_name,
        filename='{epoch:03d}',
        every_n_epochs=10,
        save_top_k=-1,
        verbose=True,
        save_on_train_epoch_end=True,
        save_last=True,
    )
    
    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()
    
    # Define the trainer 
    trainer = Trainer(
        accelerator='gpu',
        devices=list(range(CFG.num_gpus)),
        logger=wandb_logger,
        precision='16-mixed',
        callbacks=[checkpointing, program_bar],
        max_epochs=CFG.max_epochs,
        log_every_n_steps=10,
        fast_dev_run=False,
        default_root_dir = wandb_logger.experiment.dir,
        profiler = SimpleProfiler(dirpath=wandb_logger.experiment.dir, filename='profiler_report')
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--save_dir', type=str, required=True)
    args, opts = parser.parse_known_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    CFG.load(args.config)
    CFG.update(opts)
    CFG.update({'pretrained_ckpt': args.pretrained_ckpt})
    print(CFG)
    
    train(args)
    
    
    

