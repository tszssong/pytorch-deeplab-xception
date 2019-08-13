CUDA_VISIBLE_DEVICES=0 python testbatch.py --backbone resnet \
 --workers 4 \
 --gpu-ids 0 --checkname deeplab-resnet \
 --dataset coco 2>&1 | tee ./log.log
