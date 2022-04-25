_base_=[
    '../base/datasets/fashion_pedia_subcategory.py',
    '../base/models/fcos_resnet50_fpn_rnn_grammar.py',
    '../base/schedulers/scheduler_step_based_fcos_1x.py']

subcategory=[1,2,3,4,5,10,11,12,32]
grammar = [(0,31),(1,31),(2,31),(3,31),(4,31),(9,31),(10,31),(11,31)]
init_parts = set() # everything is in grammar
model=dict(
    det_head=dict(
        head=dict(
            head_cfg=dict(
                num_classes=9,
                center_with_cls=False,
                focal_loss_init_parts=init_parts,
            ),
            rnn_cfg=dict(
                num_layers=3,
            ),
            grammar=grammar,
            subcategory=subcategory,
        ), 
        test_cfg=dict(
            iou_threshold=0.6), 
        num_class=9,
        center_sampling=True))

trainer = dict(
    evaluator=dict(
        dataset_name='fashionpedia',
        #subcategory=[5,32],
        subcategory=subcategory, # the  category id start from 1. For example 'shirt,blouse' => 1
        category_ind_zero_start=True,
        )
    )

dataloader_train = dict(
    dataset=dict(
        subcategory=subcategory, # the  category id start from 1. For example 'shirt,blouse' => 1
    )
)
dataloader_val = dict(
    dataset=dict(
        subcategory=subcategory, # the  category id start from 1. For example 'shirt,blouse' => 1
    )
)

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])