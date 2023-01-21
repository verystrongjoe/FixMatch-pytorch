python -m train_supervised  --wandb --project-name wm811k-supervised-230115 --proportion 1.0 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-1.0
python -m train_supervised  --wandb --project-name wm811k-supervised-230115 --proportion 0.50 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.50
python -m train_supervised  --wandb --project-name wm811k-supervised-230115 --proportion 0.25  --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.25
python -m train_supervised  --wandb --project-name wm811k-supervised-230115 --proportion 0.1 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.1
python -m train_supervised  --wandb --project-name wm811k-supervised-230115 --proportion 0.05 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.05
python -m train_supervised  - -wandb --project-name wm811k-supervised-230115 --proportion 0.01 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003 --expand-labels --seed 5 --out results/wm811k-supervised-0.01




