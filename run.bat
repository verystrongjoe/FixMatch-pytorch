SET proportion=0.05
SET pn="waferfix-all-final-epoch1500"
SET gpuno=0
arch="resnet18"


for %%n in (2) do (
    for %%d in (0.5, 0.7, 0.9) do (
        for %%l in (1, 5, 19) do (
            for %%t in (0.25, 0.5) do (
                // set CUDA_VISIBLE_DEVICES=0 & python -m train --lambda-u %%l --nm-optim adamw --epochs 1500 --tau %%t --gpus 0 --project-name %pn%  --keep --n-weaks-combinations %%n --threshold %%d --wandb  --dataset wm811k --proportion %proportion% --arch %arch% --batch-size 128 --lr 0.05 --seed 1234
                set CUDA_VISIBLE_DEVICES=0 & python -m train --lambda-u %%l --nm-optim adamw --epochs 1500 --tau %%t --gpus 0 --project-name %pn%         --n-weaks-combinations %%n --threshold %%d --wandb  --dataset wm811k --proportion %proportion% --arch %arch% --batch-size 128 --lr 0.05 --seed 1234
            )
        )
    )
)
