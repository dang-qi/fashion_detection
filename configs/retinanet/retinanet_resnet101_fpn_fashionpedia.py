_base_='./retinanet_resnet50_fpn_fashionpedia.py'

model=dict(
    backbone=dict(
        depth=101,
    )
)
