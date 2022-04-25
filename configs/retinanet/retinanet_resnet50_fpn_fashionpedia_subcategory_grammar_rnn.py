_base_=[
    '../base/datasets/fashion_pedia_subcategory.py',
    '../base/models/retinanet_resnet50_fpn_rnn_grammar.py',
    '../base/schedulers/scheduler_step_based_retinanet_1x.py']

anchor_num=9
subcategory=[1,2,3,4,5,10,11,12,32]
grammar = [(0,31),(1,31),(2,31),(3,31),(4,31),(9,31),(10,31),(11,31)]
#grammar=[(0, 28), (0, 31), (1, 31), (1, 33), (2, 31), (2, 33), (3, 31),(4, 31), (4, 32),
#         (7, 32), (9, 31),(9, 32), (10, 31), (10, 33), (11, 31), (11, 32), (11, 33)]
#exclude_parts = set([j*anchor_num+i for i in range(anchor_num) for j in set(sum(tuple(grammar),()))])
#init_parts = set(range(46*anchor_num))-exclude_parts
init_parts=set()
model=dict(
    det_head=dict(
        head=dict(
            head_cfg=dict(
                num_classes=9,
                focal_loss_init_parts=init_parts,
            ),
            rnn_cfg=dict(
                num_layers=3,
            ),
            grammar=grammar,
            subcategory=subcategory,
        ), 
    )
)

trainer = dict(
    evaluator=dict(
        dataset_name='fashionpedia',
        subcategory=[1,2,3,4,5,10,11,12,32],
        category_ind_zero_start=True,
    )
)
dataloader_train = dict(
    dataset=dict(
        subcategory=[1,2,3,4,5,10,11,12, 32], # the  category id start from 1. For example 'shirt,blouse' => 1
    )
)
dataloader_val = dict(
    dataset=dict(
        subcategory=[1,2,3,4,5,10,11,12, 32], # the  category id start from 1. For example 'shirt,blouse' => 1
    )
)

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])
