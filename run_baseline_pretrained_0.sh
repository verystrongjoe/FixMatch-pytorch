# nohup ./run_baseline_pretrained_0.sh > nohup_baseline_pretrained_0.out &
# tail -f nohup_baseline_pretrained_0.out 

l=0.217
th=0.893
mu=4
tau=0.493
n=2


for seed in 2345 1234 3456 4567 5678
do
    for prop in 0.05 0.1 0.25 0.50 1.0
    do
        CUDA_VISIBLE_DEVICES=1 python -m train --limit-unlabled $limit_unlabled --aug_types $aug_types --mu $mu --lambda-u $l --nm-optim sgd --epochs $epoch --tau $tau --gpus 1 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $prop --arch $arch --batch-size 256 --lr $lr --seed $seed 
    done
done


# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 2 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 5 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 3 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
