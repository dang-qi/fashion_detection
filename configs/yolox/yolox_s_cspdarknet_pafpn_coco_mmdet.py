_base_=[
    '../base/datasets/coco_yolox_mmdet.py',
    #'../base/datasets/coco.py',
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
    use_amp=True,
    ema_cfg=dict(decay=0.9999),
    log_print_iter=1000,
    save_epoch_interval=10,
    num_last_epoch=15,
    scheduler=dict(iter_per_epoch=None))

lr_config=dict(
    element_lr=0.01/64,
)

dataloader_train = dict(
    num_workers=0, 
    persistent_workers=False,
)