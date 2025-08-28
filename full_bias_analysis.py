import json
import argparse
from pathlib import Path
import re
import os
import sys
from utils import *

sys.path.append("refusal_direction")

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base

sys.path.append("BBQ")
number_of_tokens_list = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def format_race(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}\n{example['context']}"

def format_arc(example: dict) -> str:
    return f"{example['context']}{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def format_qonly(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def find_biased_answer(gen_str, possible_answers):
    # Look for *exactly one* member name, ignoring spaces/punctuation around it
    chosen = []
    for ans in possible_answers:
        ans_lower = ans.lower()
        pattern = rf"(?<![a-z]){re.escape(ans_lower)}(?![a-z])"
        if re.search(pattern, gen_str.lower()):
            chosen.append(ans)

    if len(chosen) == 1:
        return chosen[0].strip()
    return None

def process_file_new(path, out_dir, model_base, model_alias):
    model = model_base.model
    tokenizer = model_base.tokenizer
    device = model.device

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preds_{path.stem}.jsonl"
    with path.open() as f_in, out_path.open('w') as f_out:
        for line in tqdm.tqdm(f_in, desc=f"Processing {path.name}"):
            ex = json.loads(line)
            
            for number_of_tokens in number_of_tokens_list:
                race_format = format_race(ex)
                arc_format = format_arc(ex)
                qonly_format = format_qonly(ex)
                prompts = [race_format, arc_format, qonly_format]
                prompts_pad = \
                    tokenize_pad_input_list(prompts, tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
                prompts_pad = [{k: v.to(device) for k, v in prompt.items()} for prompt in prompts_pad]
                completion_list = generate_completion_list(prompts_pad, model, tokenizer)
                pred_race = completion_list[0]
                pred_arc = completion_list[1]
                pred_qonly = completion_list[2]

                print(f"k={number_of_tokens}, race prompt: {race_format}\n answer: {pred_race}")
                possible_answers = [ex["ans0"], ex["ans1"], ex["ans2"]]

                ex[f"pad_{number_of_tokens}_{model_alias}_pred_race"] = find_biased_answer(pred_race, possible_answers)
                ex[f"pad_{number_of_tokens}_{model_alias}_pred_arc"] = find_biased_answer(pred_arc, possible_answers)
                ex[f"pad_{number_of_tokens}_{model_alias}_pred_qonly"] = find_biased_answer(pred_qonly, possible_answers)

            f_out.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run BBQ with Llama2 model")
    parser.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct', help='HF model name')
    parser.add_argument('--data_dir', default='BBQ/data', help='Directory with BBQ jsonl files')
    parser.add_argument('--out_dir', default='BBQ/results/', help='Where to write predictions')
    parser.add_argument('--device', type=int, default=-1, help='GPU id or -1 for cpu')
    parser.add_argument('--category_num', type=int, default=0, help='Category number to process')

    args = parser.parse_args()

    model_path = args.model_path
    category_num = args.category_num

    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)

    data_dir = Path(args.data_dir)
    out_dir = Path(os.path.join(args.out_dir, model_alias))

    category_paths = sorted(data_dir.glob('*.jsonl'))
    path = category_paths[category_num]
    print(f"Processing file: {path.name}")
    process_file_new(path, out_dir, model_base, model_alias)

if __name__ == '__main__':
    main()
