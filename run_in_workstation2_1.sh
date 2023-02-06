proportion=0.1

for n in 2 3 4
do 
    CUDA_VISIBLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --keep --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256  --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --n-weaks-combinations $n --threshold 0.7 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --keep --n-weaks-combinations $n --threshold 0.9 --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    CUDA_VISIBLE_DEVICES="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf" python -m train --epochs 250 --gpus 0 --project-name waferfix-230206 --n-weaks-combinations $n --threshold 0.9  --wandb  --dataset wm811k --proportion $proportion --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
done