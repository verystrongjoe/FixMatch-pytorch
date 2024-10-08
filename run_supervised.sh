# nohup ./run_supervised.sh > nohup_supervisd.out &
# tail -f nohup_supervisd.out 


# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 1.0  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-1.0
# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 0.50 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.50
# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 0.25 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.2
# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 0.1  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.1
# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 0.05 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.05
# CUDA_VISIBLE_DEVICES=0 python -m train_supervised --epochs 300 --gpus 1 --wandb --project-name waferfix-seed-check2 --proportion 0.01 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.01


CUDA_VISIBLE_DEVICES=0 python -m train_supervised --wandb --epochs 300 --project-name waferfix-rebuttal-supervised  --gpus 0 --proportion 1.0  --dataset wm811k --arch densenet121-3 --batch-size 128 --lr 0.003 --expand-labels --seed 5 --out results/rebuttal-densenet121-wm811k-supervised-1.0
CUDA_VISIBLE_DEVICES=0 python -m train_supervised --wandb --epochs 300 --project-name waferfix-rebuttal-supervised  --gpus 0 --proportion 0.5 --dataset wm811k --arch densenet121-3 --batch-size 128 --lr 0.003 --expand-labels --seed 5 --out results/rebuttal-densenet121-wm811k-supervised-0.5

