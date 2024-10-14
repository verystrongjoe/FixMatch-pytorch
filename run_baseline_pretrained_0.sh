# nohup ./run_baseline_pretrained_0.sh > nohup_baseline_pretrained_0.out &
# tail -f nohup_baseline_pretrained_0.out 


for p in 0.05 0.1 0.25
do
    for k in 1 2 3 5
    do
        CUDA_VISIBLE_DEVICES=0 python -m train_baseline_pretrained --K $k --project-name waferfix_baseline_pretrained_densenet121 --proportion $p --arch densenet121-3 --batch-size 128 --lr 0.003  --seed 5
    done
done