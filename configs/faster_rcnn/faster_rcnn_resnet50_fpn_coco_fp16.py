_base_=[
    '../base/datasets/coco_frcnn.py',
    '../base/models/faster_rcnn_resnet50_fpn.py',
    #'../base/schedulers/scheduler_step_based_1x.py',
    '../base/schedulers/scheduler_epoch_based_faster_rcnn_1x.py']

model=dict(
    roi_head=dict(
        box_head=dict(
            class_num=80,), 
        class_num=80,
        )
    )

trainer = dict(evaluator=dict(dataset_name='coco'),use_amp=True)
dataloader_train=dict(num_workers=3)

lr_config=dict(
    element_lr=0.02/16,
    )