# nohup ./run.sh > nohup1.out &
# tail -f nohup.out 

epoch=2000
lr=0.025
arch=resnet18
proportion=0.05

pn=wm0-$arch-lr-$lr-prop-$proportion-epoch-$epoch

for n in 6
do 
    for th in 0.9 0.95
    do
        for l in 0.5 1 5
        do 
            for t in 0.3
            do
                for m in 5
                do
                    CUDA_VISIBLE_DEVICES=0 python -m train --mu $m --lambda-u $l --nm-optim adamw --epochs $epoch --tau $t --gpus 1 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch $arch --batch-size 256 --lr $lr --seed 1234
                    CUDA_VISIBLE_DEVICES=0 python -m train --mu $m --lambda-u $l --nm-optim adamw --epochs $epoch --tau $t --gpus 1 --project-name $pn         --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch $arch --batch-size 256 --lr $lr --seed 1234
                done
            done
        done
    done
done