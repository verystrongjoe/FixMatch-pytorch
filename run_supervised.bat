python -m train_supervised --project-name temp  --gpus 0 --proportion 0.50 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.50
python -m train_supervised --project-name temp  --gpus 0 --proportion 0.25 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.25
python -m train_supervised --project-name temp  --gpus 0 --proportion 0.1  --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.1
python -m train_supervised --project-name temp  --gpus 0 --proportion 0.05 --dataset wm811k --arch resnet18 --batch-size 256 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.05

