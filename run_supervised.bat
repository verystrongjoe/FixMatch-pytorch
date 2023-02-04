python -m train_supervised  --wandb --project-name wm811k-supervised-230204  --gpus 0 --proportion 1.0  --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003  --seed 1234
python -m train_supervised  --wandb --project-name wm811k-supervised-230204  --gpus 0 --proportion 0.50 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003  --seed 1234
python -m train_supervised  --wandb --project-name wm811k-supervised-230204  --gpus 0 --proportion 0.25 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003  --seed 1234
python -m train_supervised  --wandb --project-name wm811k-supervised-230204  --gpus 0 --proportion 0.1  --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003  --seed 1234
python -m train_supervised  --wandb --project-name wm811k-supervised-230204  --gpus 0 --proportion 0.05 --dataset wm811k --arch wideresnet --batch-size 64 --lr 0.003  --seed 1234

