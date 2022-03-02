_base_=[
    '../base/datasets/fashion_pedia_frcnn.py',
    '../base/models/retinanet_resnet50_fpn_rnn_grammar.py',
    '../base/schedulers/scheduler_step_based_retinanet_1x.py']

grammar=[(0, 28), (0, 31), (1, 31), (1, 33), (2, 31), (2, 33), (3, 31),(4, 31), (4, 32),
         (7, 32), (9, 31),(9, 32), (10, 31), (10, 33), (11, 31), (11, 32), (11, 33)]
init_parts = set(range(46))-set(sum(tuple(grammar),()))
model=dict(
    det_head=dict(
        head=dict(
            head_cfg=dict(
                num_classes=46,
            ),
            rnn_cfg=dict(
                num_layers=3,
            ),
            grammar=grammar
        ), 
    )
)

trainer = dict(evaluator=dict(dataset_name='fashionpedia'))

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])
