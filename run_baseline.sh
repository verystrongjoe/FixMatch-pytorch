# nohup ./run_baseline.sh > nohup_baseline.out &
# tail -f nohup_baseline.out 


# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 1.0  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.50 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.25 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.1  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.01 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5

CUDA_VISIBLE_DEVICES=1 python -m train_baseline --K 1 --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=1 python -m train_baseline --K 2 --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=1 python -m train_baseline --K 5 --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=1 python -m train_baseline --K 3 --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
