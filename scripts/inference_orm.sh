#!/usr/bin/env bash

python src/inference_orm.py \
  --model_name base-model \
  --device cuda \
  --input_jsonl /path/to/your/input.jsonl \
  --output_jsonl /path/to/output.jsonl \
  --max_new_tokens 2048 \
  --temperature 0.6 \
  --num_return_sequences 32 \
  --do_sample
