trainer=dict(
    type='EpochBasedTrainer',
    log_print_iter=1000,
    log_save_iter=50,
    max_epoch=12,
    log_with_tensorboard=True,
    save_epoch_interval=1,
    eval_epoch_interval=1,
    accumulation_step=1,
    clip_gradient=35,
    evaluator=dict(
        type='COCOEvaluator',
        dataset_name='fashionpedia'),
    optimizer = dict(
        type = 'SGD',
        lr = 0.01,
        momentum=0.9,
        weight_decay=1e-4),

    #scheduler = dict(
    #    type = 'MultiStepLR',
    #    #milestones = [8, 11], # infer from linear lr
    #    gamma = 0.1)
    scheduler = dict(
        type = 'WarmupMultiStepLR',
        milestones = [8, 11], # infer from linear lr
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
    )
)

lr_config=dict(
    element_lr=0.01/16,
    )