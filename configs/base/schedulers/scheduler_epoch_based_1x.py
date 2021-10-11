trainer = dict(
    type='EpochBasedTrainer',
    max_epoch = 13,
    optimizer = dict(
        type = 'SGD',
        lr = 0.0025,
        momentom=0.9,
        weight_decay=1e-4),
    scheduler = dict(
        type = 'MultiStepLR',
        milestones = [8, 11],
        gamma = 0.1)
)