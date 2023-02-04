for /L %%n in (2,3,4,5,6) do (
	for /L %%t in (0.7, 0.9) do (
		for /L %%p in (0.5) do ( 
            python -m train --project-name waferfix-230204 --keep --n-weaks-combinations %%n --threshold %%t  --wandb  --dataset wm811k --proportion %%p --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234
            python -m train --project-name waferfix-230204 --n-weaks-combinations %%n --threshold %%t  --wandb  --dataset wm811k --proportion %%p --arch wideresnet --batch-size 128  --lr 0.003 --seed 1234
        )
    )
)