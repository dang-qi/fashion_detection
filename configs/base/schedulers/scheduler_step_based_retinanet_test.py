trainer=dict(
    type='StepBasedTrainer',
    log_print_iter=1,
    log_save_iter=1,
    log_with_tensorboard=True,
    save_step_interval=10000,
    eval_step_interval=10000,
    empty_cache_iter=1000,
    accumulation_step=1,
    clip_gradient=None,
    log_memory=True,
    evaluator=dict(
        type='COCOEvaluator',
        dataset_name='fashionpedia'),
    optimizer = dict(
        type = 'SGD',
        lr = 0.0025,
        momentum=0.9,
        weight_decay=1e-4),

    #scheduler = dict(
    #    type = 'MultiStepLR',
    #    #milestones = [8, 11], # infer from linear lr
    #    gamma = 0.1)
    scheduler = dict(
        type = 'WarmupMultiStepLR',
        #milestones = [8, 11], # infer from linear lr
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
    )
)

lr_config=dict(
    element_lr=0.01/16,
    element_step=8,
    milestones_split=[2/3, 8/9])