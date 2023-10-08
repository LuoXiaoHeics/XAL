# run the demo codes for ME baseline in RTE

for ini in 0 1 2;
do
    for s in 1;
    do
       python3 run_active_baselines.py \
            --data_dir=/dataset/ \
            --model_type=flan_t5 \
            --model_name_or_path=flan-t5-large \
            --do_lower_case \
            --do_test \
            --do_train \
            --max_seq_length=128 \
            --train_batch_size=1 \
            --num_train_epochs=10 \
            --eval_batch_size=12 \
            --seed=${s} \
            --learning_rate=1e-4 \
            --gpu_id=0 \
            --task=rte \
            --initial=$ini \
            --reason_seq_length=60 \
            --output_dir=output_rte/ME_  \
            --method=max_entropy
    done
done