trainer=dict(
    type='StepBasedTrainer',
    log_print_iter=1000,
    save_step_interval=10000,
    eval_step_interval=10000,
    accumulation_step=1,
    evaluator=dict(
        type='COCOEvaluator',
        dataset_name='fashionpedia'),
    optimizer = dict(
        type = 'SGD',
        lr = 0.0025,
        momentum=0.9,
        weight_decay=1e-4),

    scheduler = dict(
        type = 'MultiStepLR',
        #milestones = [8, 11], # infer from linear lr
        gamma = 0.1)
)

lr_config=dict(
    element_lr=0.00125,
    element_step=1440000,
    milestones_split=[2/3, 8/9])