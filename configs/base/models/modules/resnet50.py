backbone = dict(type='ResNet',
                depth=50,
                returned_layers=[0,1,2,3],
                init_cfg=dict(type='pretrained'))
