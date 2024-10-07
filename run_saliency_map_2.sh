# nohup ./run_saliency_map_2.sh > nohup_run_saliency_map_2.out &
# tail -f nohup_run_saliency_map.out 


CUDA_VISIBLE_DEVICES=1 python -m get_saliencymap --proportion 0.25 --arch resnet18 --batch-size 512  --lr 0.003 --seed 1234

CUDA_VISIBLE_DEVICES=1 python -m get_saliencymap --proportion 0.1  --arch resnet18 --batch-size 512  --lr 0.003 --seed 1234

CUDA_VISIBLE_DEVICES=1 python -m get_saliencymap --proportion 0.05 --arch resnet18 --batch-size 512  --lr 0.003 --seed 1234

