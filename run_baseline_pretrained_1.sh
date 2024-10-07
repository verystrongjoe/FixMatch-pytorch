# nohup ./run_baseline_pretrained_1.sh > nohup_baseline_pretrained_1.out &
# tail -f nohup_baseline_pretrained_1.out 


for p in 0.5 1.0
do
    for k in 1 2 3 5
    do
        CUDA_VISIBLE_DEVICES=1 python -m train_baseline_pretrained --K $k --project-name waferfix_baseline_pretrained_densenet121 --proportion $p --arch densenet121-3 --batch-size 256 --lr 0.003  --seed 5
    done
done