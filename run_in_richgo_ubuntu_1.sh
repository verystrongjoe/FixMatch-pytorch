proportion=0.05

for n in 2 3 4
do
  for th in 0.7 0.9
  do
    for t in 0.2 0.5
    do
      CUDA_VISIBLE_DEVICES=0 python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --keep --n-weaks-combinations $n --tau $t --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
      CUDA_VISIBLE_DEVICES=0 python -m train --epochs 250 --gpus 0 --project-name waferfix-230206        --n-weaks-combinations $n --tau $t --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    done
  done
done