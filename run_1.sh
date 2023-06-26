# nohup ./run_0.sh > nohup0.out &
# tail -f nohup0.out 

epoch=180
lr=0.01
arch=resnet18
limit_unlabled=300000
aug_types="crop, cutout, noise, rotate, shift"                   # 'crop','cutout','noise','rotate','shift'

l=0.217
th=0.893
mu=4
tau=0.493
seed=2222

pn_main=wm_230626

for n in 3 4 5
do
    for prop in 0.05 0.1 0.25 0.50 1.0
    do
        pn=$pn_main-$arch-lr-$lr-l-$l-th-$th-mu-$mu-tau-$tau-n-$n
        CUDA_VISIBLE_DEVICES=0 python -m train --limit-unlabled $limit_unlabled --aug_types $aug_types --mu $mu --lambda-u $l --nm-optim sgd --epochs $epoch --tau $tau --gpus 1 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $prop --arch $arch --batch-size 256 --lr $lr --seed $seed 
    done
done