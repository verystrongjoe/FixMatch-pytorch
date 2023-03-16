# nohup ./run_supervised_1.sh > nohup_supervisd2.out &
# tail -f nohup_supervisd2.out 

CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 1.0  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-1.0
CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 0.50 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.5
CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 0.25 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.25
# CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 0.1  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.1
CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 0.05 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.05
CUDA_VISIBLE_DEVICES=1 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check --proportion 0.01 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.01



