#!/bin/bash

python src/data_generation.py \
    --base_model model/repo \
    --dataset_path path/to/data.json \
    --save_dir /path/to/save/directory/ \
    --save_name path/to/save/name.jsonl \
    --temperature 0.6 \
    --max_length 2048 \
    --num_return_sequences number/of/samples \
    --do_sample \
    --cache_dir /home/rtha0021/wj84_scratch/ramya/.cache/