optimizer = dict(
    type = 'SGD',
    lr = 0.0025,
    n_iter = 26,
    weight_decay=1e-4)

scheduler = dict(
    type = 'multi_step',
    milestones = [16, 22],
    gamma = 0.1)