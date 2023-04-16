# nohup ./run_baseline_pretrained.sh > nohup_baseline_pretrained.out &
# tail -f nohup_baseline_pretrained.out 

# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 1.0  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 0.50 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 0.25 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 0.1  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --project-name waferfix_baseline_pretrained --proportion 0.01 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5

CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 1 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 2 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 5 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 3 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
