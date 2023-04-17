# nohup ./run_baseline_pretrained.sh > nohup_baseline_pretrained.out &
# tail -f nohup_baseline_pretrained.out 


for p in 0.05 # 0.1 0.25 0.5 1.0
do
    for k in 1 2 3 5
    do
        CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K $k --project-name waferfix_baseline_pretrained2 --proportion $p --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
    done
done


# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 2 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 5 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
# CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K 3 --project-name waferfix_baseline_pretrained --proportion 0.05 --arch resnet18 --batch-size 256 --lr 0.003  --seed 5
