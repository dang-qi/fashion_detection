import sys
import os
import argparse

import numpy as np
from torchcore.data.sampler import distributed_sampler_wrapper
from torchcore.tools import Logger

from pprint import pprint

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import datetime

from PIL.ImageDraw import Draw
from torchcore.util import Config

#from rcnn_config import config
#from tools import torch_tools
#from data import data_feeder

from torchcore.data.datasets import ModanetDataset, ModanetHumanDataset, COCOPersonDataset, COCODataset, COCOTorchVisionDataset
from torchcore.data.datasets.fashion_pedia import FashionPediaDataset
#from rcnn_dnn.networks import networks
#from rcnn_dnn.data.collate_fn import collate_fn, CollateFnRCNN
#from torchcore.data.collate import CollateFnRCNN, collate_fn_torchvision

import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as transforms
import torch.optim as optim

#from torchcore.dnn import trainer,DistributedTrainer
from torchcore.dnn.networks.faster_rcnn_fpn import FasterRCNNFPN
from torchcore.dnn.networks.roi_net import RoINet
from torchcore.dnn.networks.rpn import MyAnchorGenerator, MyRegionProposalNetwork
from torchcore.dnn.networks.heads import RPNHead
from torchcore.dnn import networks
from torchcore.dnn.networks.tools.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchcore.engine.launch import launch
import torch.distributed as dist
from torchcore.evaluation import COCOEvaluator
from torchcore.dnn.networks.detectors.build import build_detector
from torchcore.data.datasets.build import build_dataloader
from torchcore.dnn.trainer.build import build_trainer

from torchcore.dnn.networks.tools.load_from_mmdetection import load_mm_retinanet, load_mm_fcos, load_mm_general_model

def parse_commandline():
    parser = argparse.ArgumentParser(description="Training the Model")
    parser.add_argument('-c','--config_path',help='Configuration path', required=True)
    parser.add_argument('-b','--batch_size',help='Batch size per step per gpu', required=True, type=int)
    parser.add_argument('-a','--accumulation_step',help='Accumulate size', required=False, default=1, type=int)
    parser.add_argument('--gpu_num',help='gpu used to train', required=False, default=1, type=int)
    parser.add_argument('--machine_num',help='machine used to train', required=False, default=1, type=int)
    parser.add_argument('-t','--tag',help='Model tag', required=False)
    parser.add_argument('--resume',help='resume the model', action='store_true', required=False)
    parser.add_argument('--load_model_path',help='load weights for the model', default=None, required=False)
    parser.add_argument('--evaluate',help='Do you just want to evaluate the model', action='store_true', required=False)
    #parser.add_argument('--dataset', help='The dataset we are going to use', default='coco_person')
    parser.add_argument('--linear_lr', help='do we change lr linearly according to batch size', action='store_true')
    parser.add_argument('--api',help='api token for log', required=False, default=None)
    parser.add_argument('--load_from_mm_model', help='init weight from a mm detection model', action='store_true', required=False )
    parser.add_argument('--load_from_mm_model_train', help='init weight from a mm detection model', action='store_true', required=False )
    #parser.add_argument('--torchvision_model', help='Do we want to use torchvision model', action='store_true')
    #parser.add_argument('-g','--gpu',help='GPU Index', default='0')
    #parser.add_argument('--datasetpath',help='Path to the dataset',required=True)
    #parser.add_argument('--projectpath',help='Path to the project',required=True)
    return parser.parse_args()


def get_absolute_box(human_box, box):
    #box[2]+=int(human_box[0])+box[0]
    #box[3]+=int(human_box[1])+box[1]
    box[0]+=int(human_box[0])
    box[1]+=int(human_box[1])
    return box


def load_checkpoint(model, path, device, to_print=True):
    #checkpoint = torch.load(path)
    state_dict_ = torch.load(path, map_location=device)['model_state_dict']
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model.load_state_dict(state_dict, strict=True )
    #self._epoch = checkpoint['epoch']
    #self._model.load_state_dict(checkpoint['model_state_dict'])
    #self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if to_print:
        print('Chekpoint has been loaded from {}'.format(path))


def update_linear_lr(trainer_cfg,lr_cfg, batch_size ):
    accumulation_step = trainer_cfg.accumulation_step
    lr = lr_cfg.element_lr * batch_size * accumulation_step
    if 'EpochBasedTrainer' in trainer_cfg.type:
        trainer_cfg.optimizer.lr = lr
        return {}
    elif 'StepBasedTrainer' in trainer_cfg.type:
        max_step = lr_cfg.element_step // batch_size
        #milestones = [int(part*max_step) for part in lr_cfg.milestones_split]
        trainer_cfg.optimizer.lr = lr
        #trainer_cfg.scheduler.milestones=milestones
        return dict(max_step=max_step)
    else:
        raise ValueError('Unknown trainer type: {}'.format(trainer_cfg.type))


def run(args) :
    world_size = args.gpu_num * args.machine_num
    distributed = world_size>1
    world_batch_size = world_size * args.batch_size
    if args.gpu_num == 1:
        rank = 0
    else:
        rank = dist.get_rank()

    config_path = args.config_path
    cfg = Config.fromfile(config_path)
    tag = args.tag
    api_token = args.api
    batch_size_per_gpu_per_accumulation = args.batch_size

    project_path = os.path.expanduser('~/Vision/data')
    project_name = 'yolox'
    cfg.initialize_project(project_name, project_path, tag=tag)
    extra_init={}
    cfg.merge_args( args )
    if args.linear_lr:
        extra_init = update_linear_lr(cfg.trainer, cfg.lr_config, world_batch_size)
    extra_init['log_api_token'] = api_token

    #cfg.update_lr(world_batch_size)
    #cfg.resume = args.resume
    #cfg.out_feature_num = 256
    #cfg.accumulation_step = args.accumulation_step
    #cfg.nms_thresh = 0.5
    #cfg.batch_size = args.batch_size
    #cfg.optimizer.lr = cfg.optimizer.lr / args.accumulation_step
    #cfg.min_size = (640, 672, 704, 736, 768, 800)
    #max_size = 1024
    #cfg.min_size = max_size
    #cfg.max_size = max_size

    #if args.lr is not None:
    #    cfg.optimizer.lr = args.lr

    #set the paths to save all the results (model, val result)
    #cfg.build_path( params['tag'], args.dataset, model_hash='frcnn' )
    if rank == 0:
        print(cfg.pretty_text)
    cfg.dump(cfg.path_config.config_path)

    #collate_fn_rcnn = CollateFnRCNN(min_size=416, max_size=416)
    train_dataset_loader = build_dataloader(cfg.dataloader_train, distributed)
    val_dataset_loader = build_dataloader(cfg.dataloader_val,distributed=False)


    model = build_detector(cfg.model)
    if hasattr(model, 'init_weights'):
        model.init_weights()
    if args.load_from_mm_model_train:
        checkpoint=None
        mm_config = 'mmconfigs/retinanet/retinanet_r50_fpn_1x_coco.py'
        load_mm_retinanet(checkpoint, mm_config, model)
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    else:
        model = model

    evaluator = COCOEvaluator(dataset_name=cfg.dataset_name, evaluate_type=['bbox'])

    trainer = build_trainer(cfg.trainer, 
        default_args=dict(
            model=model,
            trainset=train_dataset_loader,
            testset = val_dataset_loader,
            rank=rank,
            world_size=world_size,
            path_config=cfg.path_config,
            tag=tag,
            evaluator=evaluator,
            **extra_init
    ))

    if args.resume:
        new_lr = None
        if args.linear_lr:
            new_lr = cfg.trainer.optimizer.lr
        trainer.resume_training(path=args.load_model_path,new_lr=new_lr)
    #t = my_trainer( cfg, ddp_model, device, data_loader, testset=test_data_loader, dataset_name=args.dataset, train_sampler=dist_train_sampler, benchmark=None, tag=args.tag,evaluator=evaluator, epoch_based=False, eval_step_interval=10000, save_step_interval=10000, rank=rank )
    if not args.evaluate:
        trainer.train()
    else:
        if args.load_model_path is not None:
            trainer.load_checkpoint(args.load_model_path, to_print=True)
        if args.load_from_mm_model:
            #checkpoint= 'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
            #checkpoint= 'fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth'
            checkpoint = 'yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
            #mm_config = 'mmconfigs/retinanet/retinanet_r50_fpn_1x_coco.py'
            #mm_config = 'mmconfigs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'
            mm_config = 'mmconfigs/yolox/yolox_s_8x8_300e_coco.py'
            load_mm_general_model(trainer._model, mm_config,checkpoint)
            #load_mm_fcos(checkpoint, mm_config, trainer._model,change_backbone=True)
        trainer.validate()

def cleanup():
    dist.destroy_process_group()

def main(args):
    launch(run, num_gpus_per_machine=args.gpu_num, args=(args,), dist_url='auto')

if __name__=="__main__" :
    args = parse_commandline()
    main(args)
