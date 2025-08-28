import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import os
import random
import time
import sys

# Add parent directory of bias_jailbreak_21 to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Monkey-patch sys.modules to alias the nested module as top-level
import bias_jailbreak_21.refusal_direction.pipeline as pipeline
sys.modules['pipeline'] = pipeline

from bias_jailbreak_21.refusal_direction.dataset.load_dataset import load_dataset_split
from bias_jailbreak_21.refusal_direction.pipeline.config import Config
from bias_jailbreak_21.refusal_direction.pipeline.model_utils.model_factory import construct_model_base
from bias_jailbreak_21.refusal_direction.pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from bias_jailbreak_21.refusal_direction.pipeline.submodules.generate_directions import generate_directions
from bias_jailbreak_21.refusal_direction.pipeline.submodules.select_direction import select_direction
# go back to the same directory as this file
sys.path.pop()


def format_race(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}\n{example['context']}"

def format_arc(example: dict) -> str:
    return f"{example['context']}{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def format_qonly(example: dict) -> str:
    return f"{example['question']}\n(a) {example['ans0']} (b) {example['ans1']} (c) {example['ans2']}"

def load_model(model_name: str, device: int = -1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return generator

def generate_answer(generator, prompt: str, max_new_tokens: int = 20) -> str:
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]['generated_text']
    return out[len(prompt):].strip()

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

def process_file(input_path: Path, output_dir: Path, generator, model_tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"preds_{input_path.stem}.jsonl"
    with input_path.open() as f_in, out_path.open('w') as f_out:
        for line in tqdm(f_in, desc=f"Processing {input_path.name}"):
            ex = json.loads(line)

            race_format = format_race(ex)
            arc_format = format_arc(ex)
            qonly_format = format_qonly(ex)

            pred_race = generate_answer(generator, race_format)
            pred_qonly = generate_answer(generator, qonly_format)
            pred_arc = generate_answer(generator, arc_format)

            possible_answers = [ex["ans0"], ex["ans1"], ex["ans2"]]

            ex[f"{model_tag}_pred_race"] = find_biased_answer(pred_race, possible_answers)
            ex[f"{model_tag}_pred_qonly"] = find_biased_answer(pred_qonly, possible_answers)
            ex[f"{model_tag}_pred_arc"] = find_biased_answer(pred_arc, possible_answers)
            f_out.write(json.dumps(ex) + "\n")


def process_file_new(path, out_dir, model_base, hooks, cfg, model_alias):
    baseline_fwd_pre_hooks = hooks["baseline_fwd_pre_hooks"]
    baseline_fwd_hooks = hooks["baseline_fwd_hooks"]
    ablation_fwd_pre_hooks = hooks["ablation_fwd_pre_hooks"]
    ablation_fwd_hooks = hooks["ablation_fwd_hooks"]
    actadd_fwd_pre_hooks = hooks["actadd_fwd_pre_hooks"]
    actadd_fwd_hooks = hooks["actadd_fwd_hooks"]


    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preds_{path.stem}.jsonl"
    with path.open() as f_in, out_path.open('w') as f_out:
        for line in tqdm(f_in, desc=f"Processing {path.name}"):
            ex = json.loads(line)

            # TODO: maybe remove the option of refusing
            # if did not answer, then it counts as refusing
            # for each question, find the exact refusing under "answer_info", and the ans with the "unknown" value as second in the list
            
            race_format = format_race(ex)
            arc_format = format_arc(ex)
            qonly_format = format_qonly(ex)

            instructions = [{'instruction': race_format, 'category': 'race'}, {'instruction': arc_format, 'category': 'arc'}, {'instruction': qonly_format, 'category': 'qonly'}]

            start_time = time.time()
            # baseline (x)
            completion_baseline = model_base.generate_completions(instructions, fwd_pre_hooks=baseline_fwd_pre_hooks, fwd_hooks=baseline_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
            pred_race = completion_baseline[0]['response']
            pred_arc = completion_baseline[1]['response']
            pred_qonly = completion_baseline[2]['response']
            possible_answers = [ex["ans0"], ex["ans1"], ex["ans2"]]

            ex[f"baseline_{model_alias}_pred_race"] = find_biased_answer(pred_race, possible_answers)
            ex[f"baseline_{model_alias}_pred_qonly"] = find_biased_answer(pred_qonly, possible_answers)
            ex[f"baseline_{model_alias}_pred_arc"] = find_biased_answer(pred_arc, possible_answers)

            # save completions
            ex[f"baseline_{model_alias}_completions_race"] = pred_race
            ex[f"baseline_{model_alias}_completions_qonly"] = pred_qonly
            ex[f"baseline_{model_alias}_completions_arc"] = pred_arc

            # ablation (x-rr^Tx)
            completion_ablation = model_base.generate_completions(instructions, fwd_pre_hooks=ablation_fwd_pre_hooks, fwd_hooks=ablation_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
            pred_race = completion_ablation[0]['response']
            pred_arc = completion_ablation[1]['response']
            pred_qonly = completion_ablation[2]['response']

            ex[f"ablation_{model_alias}_pred_race"] = find_biased_answer(pred_race, possible_answers)
            ex[f"ablation_{model_alias}_pred_qonly"] = find_biased_answer(pred_qonly, possible_answers)
            ex[f"ablation_{model_alias}_pred_arc"] = find_biased_answer(pred_arc, possible_answers)

            # save completions
            ex[f"ablation_{model_alias}_completions_race"] = pred_race
            ex[f"ablation_{model_alias}_completions_qonly"] = pred_qonly
            ex[f"ablation_{model_alias}_completions_arc"] = pred_arc

            # actadd (x-r)
            completion_actadd = model_base.generate_completions(instructions, fwd_pre_hooks=actadd_fwd_pre_hooks, fwd_hooks=actadd_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
            pred_race = completion_actadd[0]['response']
            pred_arc = completion_actadd[1]['response']
            pred_qonly = completion_actadd[2]['response']

            ex[f"actadd_{model_alias}_pred_race"] = find_biased_answer(pred_race, possible_answers)
            ex[f"actadd_{model_alias}_pred_qonly"] = find_biased_answer(pred_qonly, possible_answers)
            ex[f"actadd_{model_alias}_pred_arc"] = find_biased_answer(pred_arc, possible_answers)

            # save completions
            ex[f"actadd_{model_alias}_completions_race"] = pred_race
            ex[f"actadd_{model_alias}_completions_qonly"] = pred_qonly
            ex[f"actadd_{model_alias}_completions_arc"] = pred_arc

            time_taken = time.time() - start_time
            print(f"Processed {path.name} in {time_taken:.2f} seconds")
            ex["processing_time"] = time_taken

            f_out.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run BBQ with Llama2 model")
    parser.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct', help='HF model name')
    parser.add_argument('--data_dir', default='data', help='Directory with BBQ jsonl files')
    parser.add_argument('--out_dir', default='results/', help='Where to write predictions')
    parser.add_argument('--device', type=int, default=-1, help='GPU id or -1 for cpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--category_num', type=int, default=0, help='Category number to process')

    args = parser.parse_args()

    model_path = args.model_path
    direction_seed = args.seed
    category_num = args.category_num

    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)

    # (1) calculate refusal direction
    random.seed(direction_seed)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
        
    
    start_time = time.time()
    candidate_refusal_direction = generate_directions(
            model_base,
            harmful_train,
            harmless_train,
            artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))
    _, layer, direction = select_direction(
            model_base,
            harmful_train,
            harmless_train,
            candidate_refusal_direction,
            artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
            kl_threshold=10.0,# increased threshold to avoid crashes
        )
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []
    hooks = {
        "baseline_fwd_pre_hooks": baseline_fwd_pre_hooks,
        "baseline_fwd_hooks": baseline_fwd_hooks,
        "ablation_fwd_pre_hooks": ablation_fwd_pre_hooks,
        "ablation_fwd_hooks": ablation_fwd_hooks,
        "actadd_fwd_pre_hooks": actadd_fwd_pre_hooks,
        "actadd_fwd_hooks": actadd_fwd_hooks,
    }
    
    time_taken = time.time() - start_time
    print(f"Generated direction hooks in {time_taken:.2f} seconds")

    # generator = load_model(args.model, device=args.device)
    # model_tag = args.model.split('/')[-1]

    data_dir = Path(args.data_dir)
    out_dir = Path(os.path.join(args.out_dir, model_alias))

    category_paths = sorted(data_dir.glob('*.jsonl'))
    path = category_paths[category_num]
    print(f"Processing file: {path.name}")
    process_file_new(path, out_dir, model_base, hooks, cfg, model_alias)

if __name__ == '__main__':
    main()
