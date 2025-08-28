import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Helper functions
# ---------------------------
def evaluate(model, tokenizer, device, prompt, max_new_tokens=2048, do_sample=True, temperature=0.6, num_return_sequences=32):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature,
    )

    # Remove the prompt portion
    input_length = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def step_rewards_cal_function(user_query, response, model, tokenizer, index):
    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": user_query,
        "response": response
    }
    good_token = '+'
    bad_token = '-'
    step_tag = '<extra_0>'
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")
    step_sep_id = tokenizer.encode(step_tag)[0]

    process = f'{data["response"]}\n\n{step_tag}'
    conversation_str = f"{data['query']} {process}"
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_ids == step_sep_id]
    return step_scores


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch generate responses and save JSONL.")
    parser.add_argument("--model_name", type=str, default="base-model", help="HF model id or local path")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to JSONL with 'question' and 'answer'")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to write JSONL results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_return_sequences", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default False unless set)")
    args = parser.parse_args()

    device = args.device

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if device.startswith("cuda") else None,
        attn_implementation="eager"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Read JSON Lines (expects columns: 'question', 'answer')
    df = pd.read_json(args.input_jsonl, lines=True)

    logic_questions = df['question'].tolist()
    logic_gt_answers = df['answer'].tolist()

    # Generate & append JSONL results
    with open(args.output_jsonl, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(logic_questions)), desc="Generating"):
            responses = evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=logic_questions[i],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences
            )

            output = {
                "question": logic_questions[i],
                "gold_answer": logic_gt_answers[i],
                "prediction": responses
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
