_base_=[
    '../base/datasets/coco.py',
    '../base/models/fcos_resnet50_fpn.py',
    '../base/schedulers/scheduler_step_based_fcos_1x.py']

model=dict(det_head=dict(head=dict(num_classes=80), num_class=80))