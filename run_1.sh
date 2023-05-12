# nohup ./run_1.sh > nohup1.out &
# tail -f nohup1.out 

epoch=150
lr=0.01
arch=resnet18
limit_unlabled=300000
proportion=0.05
aug_types="crop, cutout, noise, rotate, shift"                   # 'crop','cutout','noise','rotate','shift'


l=0.63889213
th=0.894634897
mu=3
tau=0.487873159
n=2

pn_main=wm
pn=$pn_main-$arch-lr-$lr-l-$l-th-$th-mu-$mu-tau-$tau-n-$n



for seed in 2345 1234 3456 4567 5678
do
    for prop in 0.05 0.1 0.25 0.50 1.0
    do
        CUDA_VISIBLE_DEVICES=1 python -m train --limit-unlabled $limit_unlabled --aug_types $aug_types --mu $mu --lambda-u $l --nm-optim sgd --epochs $epoch --tau $tau --gpus 1 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $prop --arch $arch --batch-size 256 --lr $lr --seed $seed 
    done
done