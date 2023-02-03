python -m get_saliencymap --proportion 0.5 

python -m get_saliencymap --project-name waferfix-230106 --n-weaks-combinations 5 --wandb  --dataset wm811k --proportion 0.5 --arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 1234 --out results/wm811k@0.5

python -m get_saliencymap --project-name waferfix-230106 --wandb --dataset wm811k --proportion 0.25--arch wideresnet --batch-size 64  --lr 0.01 --expand-labels --seed 1234 --out results/wm811k@0.25

python -m get_saliencymap --project-name waferfix-230106 --n-weaks-combinations 3 --wandb  --dataset wm811k --proportion 0.25--arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 1234 --out results/wm811k@0.25

python -m get_saliencymap --project-name waferfix-230106 --n-weaks-combinations 4 --wandb  --dataset wm811k --proportion 0.25--arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 1234 --out results/wm811k@0.25

python -m get_saliencymap --project-name waferfix-230106 --n-weaks-combinations 5 --wandb  --dataset wm811k --proportion 0.25--arch wideresnet --batch-size 64  --lr 0.003 --expand-labels --seed 1234 --out results/wm811k@0.25
