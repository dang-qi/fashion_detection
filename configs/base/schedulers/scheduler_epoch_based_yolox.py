trainer=dict(
    type='YOLOXEpochBasedTrainer',
    log_print_iter=1000,
    log_save_iter=50,
    max_epoch=300,
    log_with_tensorboard=True,
    save_epoch_interval=1,
    eval_epoch_interval=1,
    accumulation_step=1,
    clip_gradient=None,
    evaluator=dict(
        type='COCOEvaluator',
        dataset_name='fashionpedia'),

    optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    param_cfg=dict(norm_weight_decay=0.)),

    scheduler = dict(
        type='YOLOXScheduler', 
        iter_per_epoch=None,
        num_last_epochs=15,
        max_epoch=300,
        min_lr_ratio=0.05,
        warmup=True,
        warmup_iter=0,
        warmup_factor=1,
        warmup_method='exp',
        warmup_by_epoch=False,
        update_method='iter', 
        warmup_epoch=5,
    )
)

lr_config=dict(
    element_lr=0.01/64,
    )