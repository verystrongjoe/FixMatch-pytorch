proportion=0.1

for n in 2
do 
    for l in 1 5 20
    do
        for o in 'adamw' 'sgd'
        do 
            python -m train --gpus 0 --project-name waferfix-230207_ --keep --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234  --lambda-u $l --nm-optim $o
            python -m train --gpus 0 --project-name waferfix-230207_        --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234  --lambda-u $l --nm-optim $o
        done
    done
done
