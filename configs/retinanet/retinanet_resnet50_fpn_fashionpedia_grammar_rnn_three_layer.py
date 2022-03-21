_base_=[
    '../base/datasets/fashion_pedia_frcnn.py',
    '../base/models/retinanet_resnet50_fpn_rnn_grammar.py',
    '../base/schedulers/scheduler_step_based_retinanet_1x.py']

model=dict(
    det_head=dict(
        head=dict(
            head_cfg=dict(
                num_classes=46,
            ),
            rnn_cfg=dict(
                num_layers=3,
            )
        ), 
    )
)

trainer = dict(evaluator=dict(dataset_name='fashionpedia'),
               clip_gradient=35,)

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])
