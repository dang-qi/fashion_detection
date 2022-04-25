_base_=[
    '../base/datasets/fashion_pedia.py',
    '../base/models/faster_rcnn_resnet50_fpn.py',
    '../base/schedulers/scheduler_step_based_faster_rcnn_1x.py']

model=dict(
    roi_head=dict(
        box_head=dict(
            class_num=46,), 
        class_num=46,
        )
    )

trainer = dict(
    log_print_iter=100,
    evaluator=dict(dataset_name='fashionpedia'),
)

dataloader_train=dict(
    num_workers=2, 
    prefetch_factor=2,
    persistent_workers=True,
    )

lr_config=dict(
    element_lr=0.02/16,
    )