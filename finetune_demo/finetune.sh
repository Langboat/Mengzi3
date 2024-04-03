#!/bin/bash
# Experimental environment: 8 * A100

# output_dir is by date time
echo $(date +%Y%m%d-%H%M)
accelerate launch \
    --config_file ./conf/deepspeed_conf.yaml \
    --machine_rank 0 --deepspeed_multinode_launcher standard finetune.py \
    --preprocessing_num_workers 20 \
    --train_file ./example.jsonl \
    --block_size 8192 \
    --tokenizer_name Langboat/Mengzi3-13B-Base \
    --model_name_or_path  Langboat/Mengzi3-13B-Base \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --learning_rate 3e-6 \
    --num_warmup_steps 50 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --world_size 8 \
    --num_train_epochs 3 \
    --output_dir ./savings/$(date +%Y%m%d-%H%M) \
    --with_tracking \
    --report_to tensorboard \
    --report_to_dir ./logs/tracking_log/$(date +%Y%m%d-%H%M) \
    --enable_recompute \
    --checkpointing_steps 1 \
    --seed 1024