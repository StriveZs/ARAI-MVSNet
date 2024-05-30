import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np
import scipy.io
from dataloader.mvs_dataset import MVSTrainSet, MVSTestSet
import torch
from torch.utils.data import Dataset, DataLoader

from utils.preprocess import mask_depth_image, norm_image, scale_dtu_input, crop_dtu_input
import utils.io as io

import random

# fixme: create data loader
def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = MVSTrainSet(
            root_dir=cfg.DATA.TRAIN.ROOT_DIR,
            data_list=cfg.DATA.TRAIN.TRAIN_LIST,
            num_views=cfg.DATA.TRAIN.NUM_VIEW,
            depth_num=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE
        )
    elif mode == "valid":
        dataset = MVSTrainSet(
            root_dir=cfg.DATA.VAL.ROOT_DIR,
            data_list=cfg.DATA.VAL.VAL_LIST,
            num_views=cfg.DATA.VAL.NUM_VIEW,
            depth_num=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE
        )
    elif mode == "test":
        dataset = MVSTestSet(
            root_dir=cfg.DATA.TEST.ROOT_DIR,
            data_list=cfg.DATA.TEST.TEST_LIST,
            max_h=cfg.DATA.TEST.IMG_HEIGHT,
            max_w=cfg.DATA.TEST.IMG_WIDTH,
            num_views=cfg.DATA.TEST.NUM_VIEW,
            depth_num=cfg.DATA.TEST.NUM_VIRTUAL_PLANE
        )
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

    if mode == "train" or mode == "valid":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
        drop_last=True,
    )

    return data_loader
