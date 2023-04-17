# nohup ./run_baseline.sh > nohup_baseline.out &
# tail -f nohup_baseline.out 


for p in 0.05 # 0.1 0.25 0.5 1.0
do
    for k in 1 2 3 5
    do
        CUDA_VISIBLE_DEVICES=1 python -m train_baseline --K $k --project-name waferfix_baseline --proportion $p --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
    done
done


# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 1.0  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.50 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.25 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.1  --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline --project-name waferfix_baseline --proportion 0.01 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5

