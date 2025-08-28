#!/bin/bash

python src/finetune_orm.py \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --total_batch_size 64 \
  --learning_rate 5e-4 \
  --epochs 3 \
  --valid_size 100 \
  --model_path /path/to/your/model \
  --save_path /path/to/your/output.json \
  --data_path /path/to/your/data.json \
  --datasets FOLIO