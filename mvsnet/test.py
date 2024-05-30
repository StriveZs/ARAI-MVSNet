 #!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import sys
sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from config import load_cfg_from_file
from utils.io import mkdir
from utils.logger import setup_logger
from utils.checkpoint import Checkpointer
from dataset import build_data_loader
from utils.metric_logger import MetricLogger
from utils.eval_file_logger import eval_file_logger
import os

from dataset import build_data_loader
from networks.arai_mvsnet import ARAI_MVSNet
from utils.utils import dict2cuda, dict2numpy, mkdir_p, save_cameras, write_pfm

import numpy as np
import argparse, os, time, gc, cv2
from PIL import Image
import os.path as osp
from collections import *
import sys

# 设置GPU环境
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']="0" # one GPU for testing

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ARAI-MVSNet Evaluation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default=".../ARAI-MVSNet/configs/dtu.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        action='store_true',
        default=False,
        help="whether to only use cpu for test",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test_model(model,
               save_path,
               data_loader,
               folder,
               isCPU=False,
               flow1=True,
               flow2=True
               ):
    logger = logging.getLogger("ARAI-RMVSNet.test")
    model.eval()
    end = time.time()
    total_iteration = data_loader.__len__()
    path_list = []
    tim_cnt = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            scene_name = sample["scene_name"][0]
            frame_idx = sample["frame_idx"][0][0]
            scene_path = osp.join(save_path, scene_name)

            logger.info('Process data ...')
            sample_cuda = dict2cuda(sample)

            logger.info('Testing {} frame {} ...'.format(scene_name, frame_idx))
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], flow1, flow2)
            end_time = time.time()

            outputs = dict2numpy(outputs)
            del sample_cuda

            tim_cnt += (end_time - start_time)

            logger.info('Finished {}/{}, time: {:.2f}s ({:.2f}s/frame).'.format(batch_idx + 1, len(data_loader),
                                                                          end_time - start_time,
                                                                          tim_cnt / (batch_idx + 1.)))

            rgb_path = osp.join(scene_path, 'rgb')
            mkdir_p(rgb_path)
            depth_path = osp.join(scene_path, 'depth')
            mkdir_p(depth_path)
            cam_path = osp.join(scene_path, 'cam')
            mkdir_p(cam_path)
            conf_path = osp.join(scene_path, 'confidence')
            mkdir_p(conf_path)

            ref_img = sample["imgs"][0, 0].numpy().transpose(1, 2, 0) * 255
            ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
            Image.fromarray(ref_img).save(rgb_path + '/{:08d}.png'.format(frame_idx))

            cam = sample["proj_matrices"]["stage4"][0, 0].numpy()
            save_cameras(cam, cam_path + '/cam_{:08d}.txt'.format(frame_idx))

            for stage_id in range(4):
                cur_res = outputs["stage{}".format(stage_id + 1)]
                cur_dep = cur_res["depth"][0]
                cur_conf = cur_res["confidence"][0]

                write_pfm(depth_path + "/dep_{:08d}_{}.pfm".format(frame_idx, stage_id + 1), cur_dep)
                write_pfm(conf_path + '/conf_{:08d}_{}.pfm'.format(frame_idx, stage_id + 1), cur_conf)
            logger.info('Saved results for {}/{} (resolution: {})'.format(scene_name, frame_idx, cur_dep.shape))


def test(cfg, output_dir, isCPU=False):
    logger = logging.getLogger("ARAI-MVSNet.tester")
    # build model
    model = ARAI_MVSNet(stage_configs=list(map(int, cfg.MODEL.NET_CONFIGS)), lamb=cfg.MODEL.LAMB)
    if not isCPU:
        model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    logger.info("Loading model {} ...".format(cfg.TEST.WEIGHT))

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)

    else:
        checkpointer.load(None, resume=True)

    logger.info('Success!')

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test") # mode='test
    start_time = time.time()
    test_model(model,
               save_path = cfg.TEST.SAVE_PATH,
               data_loader=test_data_loader,
               folder=output_dir.split("/")[-1],
               isCPU=isCPU,
               flow1=True,
               flow2=True
               )
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert cfg.TEST.BATCH_SIZE == 1

    isCPU = args.cpu

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("ARAI-MVSNet", output_dir, prefix="test")
    if isCPU:
        logger.info("Using CPU")
    else:
        logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir, isCPU=isCPU)


if __name__ == "__main__":
    main()
