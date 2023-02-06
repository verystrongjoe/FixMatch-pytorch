proportion=0.1

for n in 5 6
do 
    CUDA_VISIBLE_DEVICES="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --keep --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --keep --n-weaks-combinations $n --threshold 0.9 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --n-weaks-combinations $n --threshold 0.9  --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
done