# Code referred from https://github.com/openreasoner/openr/blob/main/prm/code/finetune_qwen_single_gpu.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import os
from datasets import load_dataset
import argparse
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorWithPadding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--model_path", type=str, help="repo id of the base model")
    parser.add_argument("--save_path", type=str, help="path to save peft")
    parser.add_argument("--data_path", type=str, default='/path/to/your/data.json')
    parser.add_argument("--datasets", type=str, help='dataset name')

    args = parser.parse_args()

    good_token = '+'
    bad_token = '-'
    step_tag = '<extra_0>'
    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False
    )

    print(tokenizer.eos_token_id)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")
    print(f'candidate tokens: {candidate_tokens}')
    step_tag_id = tokenizer.encode(f"{step_tag}")[0]
    print('step tag ID:', step_tag_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------
    # Define preprocessing functions
    # ------------------------------
    def create_training_examples(example):
        prompt = example["input"]
        process = f'{example["reasons"]}\n\n<extra_0>'
        labels = "+" if example["reward"] == 1 else "-"
        return {"question": prompt, "process": process, "label": labels}

    def preprocess_function(raw_example):
        example = create_training_examples(raw_example)
        input = f"{example['question']} {example['process']}"
        tokenized_inputs = tokenizer(
            input,
            truncation=True,
            padding='max_length',
            max_length=2048,
        )

        def find_all_indices(lst, element):
            return [i for i, x in enumerate(lst) if x == element]

        length = len(tokenized_inputs['input_ids'])
        indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_id)
        if not indices:
            raise ValueError("No step tag found in the input.")

        step_index = indices[0]
        tokenized_inputs['labels'] = [-100] * length
        tokenized_inputs['attention_mask'] = [1] * length

        if example['label'] == '+' or example['label'] == 1:
            tokenized_inputs['labels'][step_index] = candidate_tokens[0]
        elif example['label'] == '-' or example['label'] == 0:
            tokenized_inputs['labels'][step_index] = candidate_tokens[1]
        else:
            raise ValueError(f"Label value {example['label']} is not valid.")
        tokenized_inputs['attention_mask'][step_index] = 0
        return tokenized_inputs

    # ------------------------------
    # Load dataset
    # ------------------------------
    data_path = {"train": args.data_path}
    raw_dataset = load_dataset('json', data_files=data_path)
    dataset = raw_dataset["train"].train_test_split(test_size=args.valid_size, shuffle=True, seed=42)

    print('start processing')
    tokenized_datasets = dataset.map(preprocess_function)
    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(
        ['system', 'input', 'reasoning_label', 'label', 'predicted_answer', 'reasons', 'reward']
    )
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(
        ['system', 'input', 'reasoning_label', 'label', 'predicted_answer', 'reasons', 'reward']
    )
    print('dataset processed')

    data_collator = DataCollatorWithPadding(tokenizer)

    BATCH_SIZE = args.total_batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    print(world_size)
    print(ddp)

    fp = f'bs_{args.total_batch_size}_lr_{args.learning_rate}_datasets_{args.datasets}'
    output_path = f'./prm_results_qwen_new.{fp}'

    training_args = TrainingArguments(
        output_dir=output_path,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1000,
        save_strategy="epoch",
        bf16=True,
        report_to=None,
        dataloader_num_workers=1,
        deepspeed=None,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    new_model = args.save_path
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

if __name__ == "__main__":
    main()