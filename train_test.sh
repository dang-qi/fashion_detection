#python train.py --config_path configs/fcos/fcos_resnet50_fpn_fashion_pedia_grammar3_rnn_aug_three_layer.py -b 2 --gpu_num 4 --linear_lr "$@"
python train.py --config_path configs/retinanet/retinanet_resnet50_fpn_coco_test.py -b 2 --gpu_num 1 --linear_lr "$@"
