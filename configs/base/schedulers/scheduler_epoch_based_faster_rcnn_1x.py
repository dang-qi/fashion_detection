trainer=dict(
    type='EpochBasedTrainer',
    log_print_iter=1000,
    log_save_iter=50,
    max_epoch=12,
    log_with_tensorboard=True,
    save_epoch_interval=1,
    eval_epoch_interval=1,
    accumulation_step=1,
    clip_gradient=None,
    use_amp=False,
    evaluator=dict(
        type='COCOEvaluator',
        dataset_name='coco'),

    optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    param_cfg=dict(norm_weight_decay=0.)),

    scheduler = dict(
        type='MultiStepScheduler', 
        milestones=[8,11],
        gamma=0.1,
        warmup_factor= 1/3,
        warmup_iter=500,
        warmup_method='linear',
        update_method='epoch',
    )
)

lr_config=dict(
    element_lr=0.01/16,
    )