from torchcore.data import sampler

min_size=1024
max_size=1024
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
dataset_train = dict(
    type='FashionPediaDataset',
    root='~/data/datasets/Fashionpedia', 
    anno= '~/data/annotations/fashionpedia_instances_train.pkl',
    part='train', 
    transforms=train_transforms, 
    xyxy=True, 
    debug=False, 
    torchvision_format=False, 
    add_mask=False
)

dataloader_train = dict(
    dataset=dataset_train,
    #sampler=dict(
    #    type='DistributedSampler'
    #),
    sampler=None,
    collate=dict(
        type='CollateFnRCNN',
        min_size=min_size, 
        max_size=max_size, 
        image_mean=None, 
        image_std=None, 
        resized=False
    ),
    batch_size=2, 
    shuffle=True, 
    batch_sampler=None, 
    num_workers=0, 
    pin_memory=False, 
    drop_last=False 
)