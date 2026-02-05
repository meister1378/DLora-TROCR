#!/bin/bash
export CUTLASS_PATH=/home/mango/cutlass

# A sample training script for reference
OUTPUT_DIR="output/deepspeed_test"

deepspeed --num_gpus 1 --num_nodes 1 -- train.py \
    --deepspeed ds_config.json \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --train_csv_path "data/preprocess/fold0_train.csv" \
    --valid_csv_path "data/preprocess/fold0_valid.csv" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --encoder_model_name_or_path "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" \
    --decoder_model_name_or_path "snunlp/KR-BERT-char16424" \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --learning_rate 5e-5 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --fp16 \
    --attn_implementation "flash_attention_2" \
    --generation_num_beams 10 \
    --generation_max_length 32 \
    --overwrite_output_dir 