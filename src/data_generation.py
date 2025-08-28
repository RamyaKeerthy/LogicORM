import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except: # noqa: E722
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the output")
    parser.add_argument('--save_name', type=str, required=True, help="File to save the output")
    parser.add_argument('--load_8bit', action='store_true', help="Use 8-bit loading")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--cache_dir', type=str, default="/home/rtha0021/wj84_scratch/ramya/.cache/")
    parser.add_argument('--do_sample', action='store_true', help="Inference sampling")
    parser.add_argument('--num_return_sequences', type=int, default=1)
    # parser.add_argument('--gpt_model', action='store_true', help="Is the model GPT")

    args = parser.parse_args()
    device = detect_device()
    print(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        # load_in_8bit=args.load_8bit,
        torch_dtype='auto',
        cache_dir=args.cache_dir
    )
    model.eval()

    def evaluate( prompt,
            temperature=0.1,
            do_sample=True,
            num_return_sequences=8,
            max_new_tokens=128,):
        messages = [
            {"role": "system", "content": "Given the answer is Uncertain (C), please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature
        )
        # Calculate input length for slicing
        input_length = model_inputs.input_ids.shape[1]

        # Slice generated output to exclude the prompt
        generated_ids = [output_ids[input_length:] for output_ids in generated_ids]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses


    data = load_dataset("json", data_files=args.dataset_path)
    samples = data['train']

    outputs = []

    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    tmp_dir = os.path.join(args.save_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(save_path, 'a') as f:
        for i, sample in enumerate(tqdm(samples)):
            reason_answer = evaluate(sample['question'], max_new_tokens=args.max_length, temperature=args.temperature, num_return_sequences=args.num_return_sequences, do_sample=args.do_sample)
            print(reason_answer)
            print("-"*30)

            output = {'system': '',
                      'input': sample['question'],
                      'reasoning_label': '',
                      'label': sample['answer'],
                      'output': reason_answer
                      }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            outputs.append(output)

    with open('tmp.json', 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

    # Run postprocess_generation from data_generation_gpt for reward calculations
    # df_expanded = postprocess_generation(data/generated)
