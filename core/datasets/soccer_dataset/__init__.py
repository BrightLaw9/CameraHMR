from typing import Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from ...configs import DATASET_FOLDERS, DATASET_FILES
import os

class SoccerDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:

        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.train_dataset_prepare()
        self.val_dataset = self.val_dataset_prepare()

    def train_dataset_prepare(self):
        # if self.cfg.MODEL.TYPE == 'smplx':
        #     from .dataset_train_hands import DatasetTrain
        # else:
        from .dataset_train import DatasetTrain
        is_train = True
        if self.cfg.DATASETS.DATASETS_AND_RATIOS:
            dataset_names = self.cfg.DATASETS.DATASETS_AND_RATIOS.split('_')
            dataset_list = []
            for ds in dataset_names:
                # directory to training image frames
                img_dir = DATASET_FOLDERS[ds]
                # Expects a directory containing training data (subdirectories poses, cameras with npz files)
                npz_dir = DATASET_FILES[is_train][ds]
                for filename in os.listdir(npz_dir + "poses/"):
                    if filename.endswith(".npz"):
                        poses_path = os.path.join(npz_dir + "poses/", filename)
                        cameras_path = os.path.join(npz_dir + "cameras/", filename)
                        dataset_list.append(DatasetTrain(self.cfg, ds, img_dir, filename.replace(".npz", ""), poses_path, cameras_path))

            train_ds = torch.utils.data.ConcatDataset(dataset_list)
            return train_ds
        else:
            return None

    def val_dataset_prepare(self):
        # if self.cfg.MODEL.TYPE == 'smplx':
        #     from .dataset_val_hands import DatasetVal
        # else:
        # from .dataset_val import DatasetVal
        dataset_names = self.cfg.DATASETS.VAL_DATASETS.split('_')
        from .dataset_train import DatasetTrain
        is_train = False
        for ds in dataset_names:
            img_dir = DATASET_FOLDERS[ds]
            npz_dir = DATASET_FILES[is_train][ds]
            dataset_list = []
            for filename in os.listdir(npz_dir + "poses/"):
                if filename.endswith(".npz"):
                    poses_path = os.path.join(npz_dir + "poses/", filename)
                    cameras_path = os.path.join(npz_dir + "cameras/", filename)
                    dataset_list.append(DatasetTrain(self.cfg, ds, img_dir, filename.replace(".npz", ""), poses_path, cameras_path))
        # dataset_list = [DatasetVal(self.cfg, ds, is_train=False) for ds in dataset_names]
        return dataset_list

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True, num_workers=self.cfg.GENERAL.NUM_WORKERS, prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
        return {'img': train_dataloader}

    def val_dataloader(self):
        val_dataloaders = []
        for val_ds in self.val_dataset:
            val_dataloaders.append(torch.utils.data.DataLoader(val_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return val_dataloaders

    def test_dataloader(self):
        val_dataloaders = []
        for val_ds in self.val_dataset:
            val_dataloaders.append(torch.utils.data.DataLoader(val_ds, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS))
        return val_dataloaders
