SET epoch=2000
SET lr=0.005
SET arch=resnet18
SET proportion=0.05

SET pn="waferfix-%arch%-lr-%lr%-prop-%prop%-epoch-%epoch%"


for %%n in (3) do (
    for %%d in (0.9, 0.95) do (
        for %%l in (0.5, 1, 1.5) do (
            for %%t in (0.3) do (
                for %%m in (5) do (
                    set CUDA_VISIBLE_DEVICES=0 & python -m train --mu %%m --lambda-u %%l --nm-optim sgd --epochs %epoch% --tau %%t --gpus 1 --project-name %pn%  --keep --n-weaks-combinations %%n --threshold %%d --wandb  --dataset wm811k --proportion %proportion% --arch %arch% --batch-size 256 --lr %lr% --seed 1234
                    set CUDA_VISIBLE_DEVICES=0 & python -m train --mu %%m --lambda-u %%l --nm-optim sgd --epochs %epoch% --tau %%t --gpus 1 --project-name %pn%         --n-weaks-combinations %%n --threshold %%d --wandb  --dataset wm811k --proportion %proportion% --arch %arch% --batch-size 256 --lr %lr% --seed 1234
                )
            )
        )
    )
)
