# nohup ./run_ucb_false.sh > nohup_ucb_false.out &
# tail -f nohup_ucb_false.out 

epoch=180
lr=0.01
arch=resnet18
limit_unlabled=100000
aug_types="crop, cutout, noise, rotate, shift"                   # 'crop','cutout','noise','rotate','shift'

l=0.217
th=0.893
mu=4
tau=0.493
seed=3333
pn=waferfixmatch-ucb-230917

for n in 2
do
    for prop in 0.05 0.1 0.25 0.50 1.0
    do
        CUDA_VISIBLE_DEVICES=0 python -m train --limit-unlabled $limit_unlabled --aug_types $aug_types --mu $mu --lambda-u $l --nm-optim sgd --epochs $epoch --tau $tau --project-name $pn  --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $prop --arch $arch --batch-size 256 --lr $lr --seed $seed 
    done
done