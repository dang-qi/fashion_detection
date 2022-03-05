INF=1000000
anchor_num=9
grammar=[(4, 29), (10, 37), (10, 38), (10, 39), (10, 41), (10, 43), (10, 44)]
exclude_parts = set([j*anchor_num+i for i in range(anchor_num) for j in set(sum(tuple(grammar),()))])
init_parts = set(range(46*anchor_num))-exclude_parts
model = dict(
     type = 'RetinaNet',
     backbone=dict(type='ResNet',
                depth=50,
                returned_layers=[1,2,3,4],
                norm_layer_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                frozen_stage=1,
                init_cfg=dict(type='pretrained')),
     neck = dict(type='FPN',
               #in_channels_list=[256, 512, 1024, 2048],
               in_channels_list=[0, 512, 1024, 2048],
               out_channels=256,
               extra_blocks='last_level_p6p7_on_c5',
               relu_before_p7=False),
     det_head = dict(type = 'RetinaNetHead',
               head = dict(
                    type='HeadWithGrammarRNN',
                         head_cfg=dict(
                              type='RetinaHead',
                              in_channels=256,
                              num_anchors=9, #should change if anchor generator change
                              num_classes=46,
                              focal_loss_init_parts=init_parts),
                         rnn_cfg=dict(type='ConvRNN',
                                      input_channel=5*anchor_num,
                                      output_channel=5*anchor_num,
                                      conv_config=dict(
                                          type='Conv2d',
                                          in_channels=5*anchor_num, 
                                          out_channels=5*anchor_num, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1, 
                                          dilation=1, 
                                          groups=1, 
                                          bias=True),
                                      post_net_conv_config=dict(
                                                        type='Conv2d',
                                                        in_channels=5*anchor_num, 
                                                        out_channels=5*anchor_num, 
                                                        kernel_size=3, 
                                                        stride=1, 
                                                        padding=1, 
                                                        dilation=1, 
                                                        groups=1, 
                                                        bias=True
                                                    )
                                      ), 
                         ref_scale_ind=0, 
                         grammar=grammar,
                         anchor_num=anchor_num,
                    ),
               anchor_generator=dict(
                    type='AnchorGenerator',
                    #sizes=((32,),(64,),(128,),(256,),(512,)),
                    #sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
                    sizes = tuple((x, float(x * 2 ** (1.0 / 3)), float(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
                    #scales=[8],
                    aspect_ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128],
                    round_anchor=False),

                    #strides=[4, 8, 16, 32, 64]),
               box_coder=dict(
                    type='AnchorBoxesCoder',
                    weight=[1.0, 1.0, 1.0, 1.0]),
               class_loss=dict(
                    type='MMFocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
               #class_loss=dict(
               #     type='SigmoidFocalLoss',
               #     gamma=2.0,
               #     alpha=0.25,
               #     reduction='sum'
               #     ),
               box_matcher=dict(
                    type='MaxIoUBoxMatcher',
                    high_thresh=0.5,
                    low_thresh=0.4,
                    allow_low_quality_match=True,
                    assign_all_gt_max=True,
                    keep_max_iou_in_low_quality=False
                    ),
               box_loss=dict(
                    type='L1Loss', 
                    reduction='sum',), 
                    #beta= 1.0 / 9,
               nms_thresh=0.5,
               score_thresh=0.05,
               before_nms_top_n_test=1000,
               ),
)