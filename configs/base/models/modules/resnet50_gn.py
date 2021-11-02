backbone = dict(type='ResNet',
                depth=50,
                returned_layers=[0,1,2,3],
                norm_layer_cfg=dict(type='GN', num_groups=32),
                init_cfg=dict(type='pretrained'))
