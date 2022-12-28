python -m train --project-name waferfix-221228 --wandb --dataset wm811k --proportion 1.0 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@1.0 

python -m train --project-name waferfix-221228 --wandb  --dataset wm811k --proportion 0.50 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.50

python -m train --project-name waferfix-221228 --wandb --dataset wm811k --proportion 0.25 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.25

python -m train --project-name waferfix-221228 --wandb --dataset wm811k --proportion 0.1 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.1 

python -m train --project-name waferfix-221228 --wandb --dataset wm811k --proportion 0.05 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.05

python -m train --project-name waferfix-221228 --wandb  --dataset wm811k --proportion 0.01 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.05 

python -m train --project-name waferfix-221228 --n-weaks-combinations 3 --wandb  --dataset wm811k --proportion 1.0 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.05 

python -m train --project-name waferfix-221228 --n-weaks-combinations 4 --wandb  --dataset wm811k --proportion 1.0 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 5 --out results/wm811k@0.05 
