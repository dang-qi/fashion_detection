INF=1000000
model=dict(
    type='FCOS',
    backbone=dict(type='ResNet',
                depth=50,
                returned_layers=[1,2,3,4],
                norm_layer_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True,
                frozen_stage=1,
                init_cfg=dict(type='pretrained')),
    neck = dict(type='FPN',
               in_channels_list=[0, 512, 1024, 2048],
               out_channels=256,
               extra_blocks='last_level_p6p7'),
    det_head= dict(type='FCOSHead',
               head=dict(type='FCOSFeatureHead',
                        in_channels=256,
                        num_classes=46,
                        norm_layer_cfg=dict(type="GN", num_groups=32),
                        strides=(8,16,32,64,128),
                        centerness=True,
                        center_with_cls=True,
                        norm_on_bbox=True),
               num_class=46,
               regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                               (512, INF)),
               strides=(8, 16, 32, 64, 128),
               center_sampling=False,
               center_sample_radius=1.5,
               norm_on_bbox=True,
               enable_scale=True,
               loss_cls=dict(
                   type='SigmoidFocalLoss',
                   gamma=2.0,
                   alpha=0.25,
                   reduction='mean'
                   ),
               loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
               loss_centerness=dict(
                   type='BCEWithLogitsLoss',
                   reduction='mean'
               ),
               #loss_centerness=dict(
               #    type='CrossEntropyLoss',
               #    use_sigmoid=True,
               #    loss_weight=1.0),
               norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
               test_cfg=dict(nms_pre=1000,
                             min_bbox_size=0,
                             score_thr=0.05,
                             iou_threshold=0.5,
                             #nms=dict(type='nms', iou_threshold=0.5),
                             max_per_img=100)

               )
)