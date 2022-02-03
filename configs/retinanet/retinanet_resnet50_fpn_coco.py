_base_=[
    '../base/datasets/coco_frcnn.py',
    '../base/models/retinanet_resnet50_fpn.py',
    '../base/schedulers/scheduler_step_based_retinanet_1x.py']

model=dict(
    det_head=dict(
        head=dict(
            num_classes=80,), 
        ))

trainer = dict(evaluator=dict(dataset_name='coco'))

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])