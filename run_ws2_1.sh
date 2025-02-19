proportion=0.05
epoch=200
lr=0.007
arch=alexnet

gpu_0_0="MIG-4e9bdbba-d0ea-5377-ae8a-a78ccab2f5e5"
gpu_0_0="MIG-f45e64c7-dc06-5453-a81e-9bc9ecc30588"
gpu_1_0="MIG-91fc8fce-9c3d-57d0-b652-6270d2d1d7d4"
gpu_1_1="MIG-5ceca708-e5aa-5675-b039-37a77bd4b6cf"
gpu_2_0="MIG-dc45e153-fb1e-5b2d-8a31-d3fb9494cd80"
gpu_2_1="MIG-21d343f4-de6e-5d44-9774-e2f3dbab968d"
gpu_3_0="MIG-0b2452d4-9b27-530f-a6f1-1c2d05dfaa72"
gpu_3_1="MIG-e46a8085-268f-5417-8e5a-a9e20578424d"

pn=waferfix-$arch-lr-$lr-prop-$proportion

gpuno1=$gpu_2_1
gpuno2=$gpu_3_0 

for n in 3
do 
    for th in 0.5 0.7 0.9
    do
        for l in 1 5 10
        do 
            for t in 0.3 0.5
            do
                for m in 3 10:
                do
                    CUDA_VISIBLE_DEVICES=$gpuno1 python -m train --mu $m --lambda-u $l --nm-optim adamw --epochs $epoch --tau $t --gpus 1 --project-name $pn  --keep --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch $arch --batch-size 256 --lr $lr --seed 1234
                    CUDA_VISIBLE_DEVICES=$gpuno1 python -m train --mu $m --lambda-u $l --nm-optim adamw --epochs $epoch --tau $t --gpus 1 --project-name $pn         --n-weaks-combinations $n --threshold $th --wandb  --dataset wm811k --proportion $proportion --arch $arch --batch-size 256 --lr $lr --seed 1234
                done
            done
        done
    done
done