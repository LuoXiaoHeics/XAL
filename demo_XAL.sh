# run the demo codes for XAL in RTE

for ini in 0 1 2;
do
    python3 run_active_rank.py \
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
        --seed=2 \
        --learning_rate=1e-4 \
        --gpu_id=0 \
        --lw_g=0.1 \
        --lw_r=0.01 \
        --lam=0.5 \
        --task=rte \
        --initial=$ini \
        --reason_seq_length=60 \
        --output_dir=output_rte/rank_ 
done