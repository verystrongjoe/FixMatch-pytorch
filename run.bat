for /L %%n in (2,3,4,5,6) do (
            python -m train --gpus 0 --project-name waferfix-230204 --keep --n-weaks-combinations %%n --threshold 0.7 --wandb  --dataset wm811k --proportion 0.5 --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234
            python -m train --gpus 0 --project-name waferfix-230204 --n-weaks-combinations %%n --threshold 0.7 --wandb  --dataset wm811k --proportion 0.5 --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234


            python -m train --gpus 0 --project-name waferfix-230204 --keep --n-weaks-combinations %%n --threshold 0.9 --wandb  --dataset wm811k --proportion 0.5 --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234
            python -m train --gpus 0 --project-name waferfix-230204 --n-weaks-combinations %%n --threshold 0.9  --wandb  --dataset wm811k --proportion 0.5 --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234
)