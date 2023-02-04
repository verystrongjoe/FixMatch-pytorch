#python -m get_saliencymap --project-name waferfix --proportion 1.0 --arch wideresnet --batch-size 64  --lr 0.003 --seed 1234 --out results/wm811k@0.5

python -m get_saliencymap --project-name waferfix --proportion 0.5  --arch wideresnet --batch-size 64  --lr 0.003 --seed 1234 --out results/wm811k@0.5

python -m get_saliencymap --project-name waferfix --proportion 0.25 --arch wideresnet --batch-size 64  --lr 0.003 --seed 1234 --out results/wm811k@0.25

python -m get_saliencymap --project-name waferfix --proportion 0.1  --arch wideresnet --batch-size 64  --lr 0.003 --seed 1234 --out results/wm811k@0.10

python -m get_saliencymap --project-name waferfix --proportion 0.05 --arch wideresnet --batch-size 64  --lr 0.003 --seed 1234 --out results/wm811k@0.05
