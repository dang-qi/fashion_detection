min_size=640
max_size=640
batch_size_per_gpu=2
dataset_name='coco'
train_single_transforms= None
mix_up_transforms = [dict(type='ResizeMax',
                          max_size=(min_size,min_size), #(h,w)
                          ),
                    dict(type='Padding',
                        size=(min_size,min_size), # (h,w)
                        pad_value=(114,114,114),
                        ),
                    dict(type='RandomMirror',
                        probability=0.5, 
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomAbsoluteScale',
                        low=max_size*0.8,
                        high=max_size*1.6,
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomCrop',
                        size=max_size,
                        box_inside=True, 
                        mask_key=None)
                    ]
train_transforms = [dict(type='Mosaic',
                        img_size=(min_size, min_size), # (h,w)
                        center_ratio_range=(0.5, 1.5),
                        min_bbox_size=0,
                        bbox_clip_border=True,
                        skip_filter=True,
                        pad_val=114,
                        prob=1.0,
                        return_pillow_img=True ),
                    dict(
                        type='RandomAffine',
                        scaling_ratio_range=(0.1, 2),
                        border=(-min_size // 2, -min_size // 2)),
                    dict(type='Resize',
                        size=(min_size, min_size)),
                    dict(type='MixUp',
                        transforms=mix_up_transforms,
                        min_bbox_size=5,
                        min_area_ratio=0.2,
                        max_aspect_ratio=20),
                    dict(type='HSVColorJittering',
                        h_range=5,
                        s_range=30,
                        v_range=30),
                    dict(type='RandomMirror',
                        probability=0.5, 
                        targets_box_keys=['boxes'], 
                        mask_key=None),

                    #dict(type='RandomAbsoluteScale',
                    #    low=max_size/2,
                    #    high=max_size*2,
                    #    targets_box_keys=['boxes'], 
                    #    mask_key=None),
                    #dict(type='RandomCrop',
                    #    size=max_size,
                    #    box_inside=True, 
                    #    mask_key=None)
                    ]
val_transforms = None
dataset_train = dict(
    type='MultiImageDataset',
    dataset_cfg=dict(
        type='COCODataset',
        root='~/data/datasets/COCO/train2017', 
        anno= '~/data/annotations/coco2017_instances.pkl',
        part='train2017', 
        transforms=None, 
        xyxy=True, 
        debug=False, 
        torchvision_format=False, 
        add_mask=False
    ),
    transforms=train_transforms
)

dataset_val = dict(
    type='COCODataset',
    root='~/data/datasets/COCO/val2017', 
    anno= '~/data/annotations/coco2017_instances.pkl',
    part='val2017', 
    transforms=val_transforms, 
    xyxy=True, 
    debug=False, 
    torchvision_format=False, 
    add_mask=False
)


dataloader_train = dict(
    dataset=dataset_train,
    persistent_workers=True,
    sampler=dict(
        type='RandomSampler', # No need to use distributed sample here because the distributed sampler wrapper will be used in distributed situation
        data_source=None # set data_source to None if the dataset want to be used here
    ),
    collate=dict(
        type='CollateFnPadding',
    ),
    #collate=dict(
    #    type='CollateFnRCNN',
    #    min_size=min_size, 
    #    max_size=max_size, 
    #    image_mean=None, 
    #    image_std=None, 
    #    resized=True,
    #),
    batch_size=batch_size_per_gpu, 
    batch_sampler=None, 
    num_workers=2, 
    pin_memory=False, 
    drop_last=False 
)

dataloader_val = dict(
    dataset=dataset_val,
    sampler=dict(
        type='SequentialSampler',
        data_source=None # set data_source to None if the dataset want to be used here
    ),
    collate=dict(
        type='CollateFnRCNN',
        min_size=min_size, 
        max_size=max_size, 
        image_mean=None, 
        image_std=None, 
        resized=False
    ),
    batch_size=1, 
    shuffle=False, 
    batch_sampler=None, 
    num_workers=0, 
    pin_memory=False, 
    drop_last=False 
)