
model = dict(
     type = 'FasterRCNNFPN',
     backbone=dict(type='ResNet',
                depth=50,
                returned_layers=[1,2,3,4],
                norm_layer_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                frozen_stage=1,
                init_cfg=dict(type='pretrained')),
     neck = dict(type='FPN',
               in_channels_list=[256, 512, 1024, 2048],
               out_channels=256,
               extra_blocks='last_level_max_pool'),
     rpn = dict(type = 'RPN',
               rpn_head = dict(
                    type='RPNHead',
                    in_channels=256),
                    #num_anchors=,),
               sampler=dict(
                    type='RandomSampler',
                    total_num=256,
                    pos_ratio=0.5,
                    add_gt_as_sample=False,
               ),
               anchor_generator=dict(
                    type='AnchorGenerator',
                    sizes=((32,),(64,),(128,),(256,),(512,)),
                    #scales=[8],
                    round_anchor=False,
                    aspect_ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
               bbox_coder=dict(
                    type='AnchorBoxesCoder',
                    weight=[1.0, 1.0, 1.0, 1.0]),
               box_matcher=dict(
                    type='MaxIoUBoxMatcher',
                    high_thresh=0.7, 
                    low_thresh=0.3, 
                    min_gt_ious=0.3,
                    allow_low_quality_match=True, 
                    assign_all_gt_max=True, 
                    keep_max_iou_in_low_quality=False
               ),
               class_loss=dict(
                    type='BCEWithLogitsLoss',
                    ),
               box_loss=dict(
                    type='L1Loss', 
                    reduction='sum', 
                    ),
               nms_thresh=0.7,
               pre_nms_top_n_train = 2000,
               post_nms_top_n_train = 1000,
               pre_nms_top_n_test = 1000,
               post_nms_top_n_test = 1000),
     roi_head = dict(type='RoINet',
                    box_head=dict(
                         type='FastRCNNHead',
                         class_num=46,
                         pool_w=7,
                         pool_h=7,
                         out_feature_num=256
                    ),
                    sampler=dict(
                         type='RandomSampler',
                         total_num=512,
                         pos_ratio=0.25,
                         add_gt_as_sample=True,
                    ),
                    #sampler=dict(
                    #     type='PosNegSampler',
                    #     pos_sample_num=128,
                    #     neg_sample_num=384),
                    box_matcher=dict(
                         type='MaxIoUBoxMatcher',
                         high_thresh=0.5, 
                         low_thresh=0.5, 
                         min_gt_ious=0.5,
                         allow_low_quality_match=False, 
                         assign_all_gt_max=True, 
                         keep_max_iou_in_low_quality=False
                    ),
                    roi_extractor=dict(
                         type='RoiAliagnFPN',
                         pool_h=7,
                         pool_w=7,
                         sampling=2
                    ),
                    box_coder=dict(
                         type='AnchorBoxesCoder',
                         box_code_clip=None, 
                         weight=[10.0, 10.0, 5.0, 5.0]
                    ),
                    box_loss=dict(
                         type='L1Loss',
                         reduction='sum', 
                    ),
                    class_loss=dict(
                         type='CrossEntropyLoss',
                         reduction='mean'
                    ),
                    score_thresh=0.001,
                    nms_thresh=0.5,
                    iou_low_thresh=0.5,
                    iou_high_thresh=0.5,
                    class_num=46,
                    #feature_names=['1','2','3','4'],
                    feature_names=[1,2,3,4],
                    #feature_names=[2,3,4,'pool'],
                    feature_strides=[4,8,16,32]
                    )
)