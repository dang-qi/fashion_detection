min_size=1024
max_size=1024
batch_size_per_gpu=2
dataset_name='fashionpedia'
train_transforms = [dict(type='RandomMirror',
                        probability=0.5, 
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomAbsoluteScale',
                        low=max_size/2,
                        high=max_size*2,
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomCrop',
                        size=max_size,
                        box_inside=True, 
                        mask_key=None)
                    ]
val_transforms = None
dataset_train = dict(
    type='FashionPediaDataset',
    root='~/data/datasets/Fashionpedia/train', 
    anno= '~/data/annotations/fashionpedia_instances_train.pkl',
    part='train', 
    transforms=train_transforms, 
    xyxy=True, 
    debug=False, 
    torchvision_format=False, 
    add_mask=False
)

dataset_val = dict(
    type='FashionPediaDataset',
    root='~/data/datasets/Fashionpedia/test', 
    anno= '~/data/annotations/fashionpedia_instances_val.pkl',
    part='val', 
    transforms=val_transforms, 
    xyxy=True, 
    debug=False, 
    torchvision_format=False, 
    add_mask=False
)


dataloader_train = dict(
    dataset=dataset_train,
    sampler=dict(
        type='RandomSampler', # No need to use distributed sample here because the distributed sampler wrapper will be used in distributed situation
        data_source=None # set data_source to None if the dataset want to be used here
    ),
    collate=dict(
        type='CollateFnRCNN',
        min_size=min_size, 
        max_size=max_size, 
        image_mean=None, 
        image_std=None, 
        resized=True,
    ),
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