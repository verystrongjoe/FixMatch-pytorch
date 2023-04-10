# nohup ./run_baselines.sh > nohup_baselines.out &
# tail -f nohup_baselines.out 


# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 1.0  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 0.50 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 0.25 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 0.1  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baselines --project-name waferfix-ensemble --proportion 0.01 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5


CUDA_VISIBLE_DEVICES=0 python -m train_baselines --K 1 --project-name waferfix-ensemble --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baselines --K 2 --project-name waferfix-ensemble --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baselines --K 3 --project-name waferfix-ensemble --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
CUDA_VISIBLE_DEVICES=0 python -m train_baselines --K 5 --project-name waferfix-ensemble --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
