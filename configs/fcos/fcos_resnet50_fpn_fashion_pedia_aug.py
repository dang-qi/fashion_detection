_base_=[
    '../base/datasets/fashion_pedia.py',
    '../base/models/fcos_resnet50_fpn.py',
    '../base/schedulers/scheduler_step_based_fcos_1x.py']

model=dict(
    det_head=dict(
        head=dict(
            num_classes=46, 
            center_with_cls=False), 
        test_cfg=dict(
            iou_threshold=0.6), 
        num_class=46,
        center_sampling=True))

trainer = dict(evaluator=dict(dataset_name='fashionpedia'))

lr_config=dict(
    element_lr=0.01/16,
    element_step=90000*16,
    milestones_split=[2/3, 8/9])