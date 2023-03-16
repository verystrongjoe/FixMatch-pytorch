proportion=0.05
epoch=500
lr=0.05
arch=resnet18

pn=waferfix-$arch-lr-$lr-prop-$proportion


for n in 4 
do 
    for th in 0.95
    do
        for l in 1 5 10
        do 
            for t in 0.5
            do
                CUDA_VISIBLE_DEVICES=0 python -m train --lambda-u $l --nm-optim adamw --epochs 600 --tau $t --gpus 0 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
                CUDA_VISIBLE_DEVICES=0 python -m train --lambda-u $l --nm-optim adamw --epochs 600 --tau $t --gpus 0 --project-name $pn         --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch $arch --batch-size 256 --lr 0.003 --seed 1234
            done
        done
    done
done


