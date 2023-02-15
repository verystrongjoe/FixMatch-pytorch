proportion=0.05
pn=waferfix-all-final-$proportion

for n in 2 
do 
    for th in 0.5 0.7 0.9
    do
        for l in 1 5 10
        do 
            for t in 0.25 0.5
            do
                CUDA_VISIBLE_DEVICES=0 python -m train --lambda-u $l --nm-optim adamw --epochs 600 --tau $t --gpus 0 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
                CUDA_VISIBLE_DEVICES=0 python -m train --lambda-u $l --nm-optim adamw --epochs 600 --tau $t --gpus 0 --project-name $pn         --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
            done
        done
    done
done


