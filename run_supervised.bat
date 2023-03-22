python -m train_supervised --epochs 300 --project-name waferfix-seed-check  --gpus 1 --proportion 1.0  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-1.0
python -m train_supervised --epochs 300 --project-name waferfix-seed-check  --gpus 1 --proportion 0.5 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.5
python -m train_supervised --epochs 300 --project-name waferfix-seed-check  --gpus 1 --proportion 0.25 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.25
python -m train_supervised --epochs 300 --project-name waferfix-seed-check  --gpus 1 --proportion 0.1  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.1
python -m train_supervised --epochs 300 --project-name waferfix-seed-check  --gpus 1 --proportion 0.05 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.05

