    model = dict(
        type='FasterRCNN',
        backbone=dict(
            type='Resnet',
            
        )
    )
    rpn = edict(
        nms_thresh = 0.7,
        min_box_size = 1e-3,
        min_box_size = 3,
        pre_nms_top_n_train = 2000,
        post_nms_top_n_train = 2000,
        pre_nms_top_n_test = 1000,
        post_nms_top_n_test = 1000)
    cfg.rpn = rpn

    roi_pool = edict()
    roi_pool.pool_w = 7
    roi_pool.pool_h = 7
    cfg.roi_pool = roi_pool

    roi_head = edict()
    roi_head.score_thre = 0.001
    roi_head.iou_low_thre = 0.5
    roi_head.iou_high_thre = 0.5
    roi_head.pos_sample_num = 128
    roi_head.neg_sample_num = 384
    roi_head.detection_per_image = 100
    roi_head.pool_w = 7
    roi_head.pool_h = 7
    roi_head.nms_thresh = 0.5
    roi_head.out_feature_num = 256
    roi_head.box_weight = [10.0, 10.0, 5.0, 5.0]
    cfg.roi_head = roi_head