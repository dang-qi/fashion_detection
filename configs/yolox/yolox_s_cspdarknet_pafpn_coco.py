_base_=[
    '../base/datasets/coco.py',
    '../base/models/yolox_s_cspdarknet_pafpn.py',
    '../base/schedulers/scheduler_epoch_based_yolox.py']

model=dict(
    det_head=dict(
        num_classes=80,
        head_cfg=dict(
            num_classes=80,), 
        ))

trainer = dict(
    evaluator=dict(dataset_name='coco'),
    use_amp=False,
    log_print_iter=1000,
    scheduler=dict(iter_per_epoch=None))

lr_config=dict(
    element_lr=0.01/64,
)

min_size=640
max_size=640
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
dataloader_train=dict(
    collate=dict(min_size=min_size, max_size=max_size),
    num_workers=0,)
dataloader_val=dict(collate=dict(min_size=min_size, max_size=max_size))