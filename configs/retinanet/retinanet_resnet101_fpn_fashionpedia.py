_base_='./retinanet_resnet50_fpn_fashionpedia'

model=dict(
    backbone=dict(
        depth=101,
    )
)