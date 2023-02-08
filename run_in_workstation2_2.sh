proportion=1.0
gpuno="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4"

pn=waferfix-all-final
t=0.3
l=10

for n in 2 3 4 5 6 
do 
    for th in 0.9 0.95 
    do
          CUDA_VISIBLE_DEVICES=$gpuno python -m train --lambda-u 10 --nm-optim adamw --epochs 400 --tau $t --gpus 0 --project-name $pn  --keep --n-weaks-combinations 2 --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
          CUDA_VISIBLE_DEVICES=$gpuno python -m train --lambda-u 10 --nm-optim adamw --epochs 400 --tau $t --gpus 0 --project-name $pn         --n-weaks-combinations 2 --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    done
done