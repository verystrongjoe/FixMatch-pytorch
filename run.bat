SET proportion=0.05
SET pn="waferfix-all-final"
SET gpuno=0

for %%n in (2,3,4,5,6) do (
    for %%t in (0.9, 0.95) do (
        set CUDA_VISIBLE_DEVICES=0 & python -m train --lambda-u 10 --nm-optim adamw --epochs 400 --tau 0.3 --gpus 0 --project-name %pn%  --keep --n-weaks-combinations %%n --threshold %%t --wandb  --dataset wm811k --proportion %proportion% --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
        set CUDA_VISIBLE_DEVICES=0 & python -m train --lambda-u 10 --nm-optim adamw --epochs 400 --tau 0.3 --gpus 0 --project-name %pn%         --n-weaks-combinations %%n --threshold %%t --wandb  --dataset wm811k --proportion %proportion% --arch wideresnet --batch-size 256 --lr 0.003 --seed 1234
    )
)
