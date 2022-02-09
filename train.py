#import _init
import sys
import os
import argparse
import pickle
import platform
import progressbar
import numpy as np
import datetime
from torchcore.data.sampler import distributed_sampler_wrapper
from torchcore.tools import Logger
import torchvision
import math
from torch import nn
import json
import tqdm
import time
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

#class my_trainer(DistributedTrainer):
#
#    def validate_onece(self):
#        self.validate()
#
#    #def validate(self):
#    #    if isinstance(self._model, DDP):
#    #        if not self.is_main_process():
#    #            print('one process resturn')
#    #            return
#    #    if self._cfg.torchvision_model:
#    #        self._validate_torchvision()
#    #    else:
#    #        self._validate()
#
#    def _validate( self ):
#        print('start to validate')
#        #self._model.test_mode = 'second'
#        #self._model.module.test_mode = 'second'
#        self._model.eval()
#        if isinstance(self._model, DDP):
#            test_model = self._model.module
#        else:
#            test_model = self._model
#
#        total_time = 0
#
#        results = []
#        with torch.no_grad() :
#            for idx,(inputs, targets) in enumerate(tqdm.tqdm(self._testset, 'evaluating')):
#                inputs = self._set_device( inputs )
#                start = time.time()
#                output = test_model( inputs)
#                batch_size = len(output['boxes'])
#                #for i, im in enumerate(output):
#                for i in range(batch_size):
#                    if len(output['boxes'][i]) == 0:
#                        continue
#                    # convert to xywh
#                    output['boxes'][i][:,2] -= output['boxes'][i][:,0]
#                    output['boxes'][i][:,3] -= output['boxes'][i][:,1]
#                    for j in range(len(output['boxes'][i])):
#                        results.append({'image_id':int(targets[i]['image_id']), 
#                                        'category_id':output['labels'][i][j].cpu().numpy().round(decimals=2).tolist(), 
#                                        'bbox':output['boxes'][i][j].cpu().numpy().round(decimals=2).tolist(), 
#                                        'score':output['scores'][i][j].cpu().numpy().round(decimals=2).tolist()})
#                total_time += time.time()-start
#        print('total time is {}s'.format(total_time))
#        if isinstance(self._model, DDP):
#            model_time = self._model.module.total_time
#        else:
#            model_time = self._model.total_time
#        print('total model time is {}s'.format(model_time))
#        print('total model time is {}s'.format(sum(model_time.values())))
#        print('average time per image is {} s'.format(sum(model_time.values())/len(self._testset)))
#        average_model_time = {k:v/len(self._testset) for k,v in model_time.items()}
#        print('model average time per image by part is {}'.format(average_model_time))
#                
#        result_path = '{}temp_result.json'.format(self._tag)
#        with open(result_path,'w') as f:
#            json.dump(results,f)
#        self.evaluator.evaluate(result_path)
#
#    def _validate_torchvision( self ):
#        print('start to validate')
#        self._model.eval()
#
#        results = []
#        with torch.no_grad() :
#            for idx,(inputs, targets) in enumerate(tqdm.tqdm(self._testset, 'evaluating')):
#                inputs = self._set_device( inputs )
#                #if self._cfg.torchvision_model:
#                #    output = self._model( inputs['data'])
#                #else:
#                output = self._model.module( inputs)
#                batch_size = len(output)
#                #for i, im in enumerate(output):
#                for i in range(batch_size):
#                    if len(output[i]['boxes']) == 0:
#                        continue
#                    # convert to xywh
#                    output[i]['boxes'][:,2] -= output[i]['boxes'][:,0]
#                    output[i]['boxes'][:,3] -= output[i]['boxes'][:,1]
#                    for j in range(len(output[i]['boxes'])):
#                        results.append({'image_id':int(targets[i]['image_id']), 
#                                        'category_id':output[i]['labels'][j].cpu().numpy().round(decimals=2).tolist(), 
#                                        'bbox':output[i]['boxes'][j].cpu().numpy().round(decimals=2).tolist(), 
#                                        'score':output[i]['scores'][j].cpu().numpy().round(decimals=2).tolist()})
#                #output = self._model['net']( inputs, just_embedding=True) # debug
#                #bench.update( targets, output )
#                
#        out_path = '{}_result.json'.format(self._tag)
#        with open(out_path,'w') as f:
#            json.dump(results,f)
#        self.evaluator.evaluate(out_path)
#        #self.eval_result(out_path, anno_type='bbox', dataset=self._dataset_name)
#        #self.eval_result(out_path, anno_type='segm', dataset=self._dataset_name)



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

#def get_model(cfg, torchvision_model=False):
#    if torchvision_model:
#        print('use torchvision model')
#        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=True)
#        #model = torchvision.models.detection.FasterRCNN(backbone, num_classes=cfg.class_num, min_size=cfg.min_size, max_size=cfg.max_size)
#        model = torchvision.models.detection.MaskRCNN(backbone, num_classes=cfg.class_num, min_size=cfg.min_size, max_size=cfg.max_size)
#    else:
#        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=True)
#        heads = {}
#        #anchor_generator = MyAnchorGenerator(sizes=((32,),(64,),(128,),(256,),(512,)), aspect_ratios=(0.5,1.0,2.0))
#        anchor_generator = MyAnchorGenerator(sizes=((16,),(32,),(64,),(128,),(256,)), aspect_ratios=(0.5,1.0,2.0))
#        #anchor_generator = MyAnchorGenerator(sizes=((32,),(64,),(128,),(256,)), aspect_ratios=(0.5,1.0,2.0))
#        num_anchors_per_location = anchor_generator.num_anchors_per_location()
#        assert all(num_anchors_per_location[0] == item for item in num_anchors_per_location)
#        #print('num anchors per location',num_anchors)
#        rpn_head = RPNHead(256, num_anchors=num_anchors_per_location[0])
#        rpn = MyRegionProposalNetwork(anchor_generator, rpn_head, cfg.rpn)
#        roi_head = RoINet(cfg.roi_head)
#        heads['rpn'] = rpn
#        heads['bbox'] = roi_head
#        #parts = ['heatmap', 'offset', 'width_height']
#        model = FasterRCNNFPN(backbone, heads=heads, cfg=cfg, training=True, debug_time=True )
#    return model

def update_linear_lr(trainer_cfg,lr_cfg, batch_size ):
    accumulation_step = trainer_cfg.accumulation_step
    lr = lr_cfg.element_lr * batch_size * accumulation_step
    if trainer_cfg.type=='EpochBasedTrainer':
        trainer_cfg.optimizer.lr = lr
        return {}
    elif trainer_cfg.type=='StepBasedTrainer':
        max_step = lr_cfg.element_step // batch_size
        milestones = [int(part*max_step) for part in lr_cfg.milestones_split]
        trainer_cfg.optimizer.lr = lr
        trainer_cfg.scheduler.milestones=milestones
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
    project_name = 'retinanet'
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
        trainer.resume_training()
    #t = my_trainer( cfg, ddp_model, device, data_loader, testset=test_data_loader, dataset_name=args.dataset, train_sampler=dist_train_sampler, benchmark=None, tag=args.tag,evaluator=evaluator, epoch_based=False, eval_step_interval=10000, save_step_interval=10000, rank=rank )
    if not args.evaluate:
        trainer.train()
    else:
        if args.load_model_path is not None:
            trainer.load_checkpoint(args.load_model_path, to_print=True)
        trainer.validate()

def cleanup():
    dist.destroy_process_group()

def main(args):
    launch(run, num_gpus_per_machine=args.gpu_num, args=(args,), dist_url='auto')

if __name__=="__main__" :
    args = parse_commandline()
    main(args)
