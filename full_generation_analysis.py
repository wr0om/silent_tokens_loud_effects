from utils import *


# pip install -U transformers accelerate torch datasets evaluate sacrebleu bert-score
# pip install -U git+https://github.com/neulab/BARTScore.git


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import evaluate
import sys
sys.path.append("BARTScore")  # harmless if already installed as a package
from bart_score import BARTScorer


sys.path.append("refusal_direction")
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base


# ======================
# Config
# ======================
DATASET_NAME = "truthfulqa"  # choices: "truthfulqa"
N_EXAMPLES = 128           # increase once wiring works
SEED = 42
number_of_tokens_list = [0, 1, 2, 4, 8, 16, 32, 64, 128]

# BERTScore settings
BERTSCORE_LANG = "en"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_RESCALE = True

# BARTScore settings
BARTSCORE_CHECKPOINT = "facebook/bart-large-cnn"
BARTSCORE_BATCH = 4

random.seed(SEED)


# ======================
# Metric helpers
# ======================

# https://aclanthology.org/P02-1040.pdf?utm_source=chatgpt.com
def compute_bleu(preds: List[str], refs: List[str]) -> float:
    sbleu = evaluate.load("sacrebleu")
    return float(sbleu.compute(predictions=preds, references=[[r] for r in refs])["score"])

# https://openreview.net/forum?id=SkeHuCVFDr&utm_source=chatgpt.com
def compute_bertscore_f1(preds: List[str], refs: List[str]) -> float:
    bsc = evaluate.load("bertscore")
    res = bsc.compute(
        predictions=preds,
        references=refs,
        lang=BERTSCORE_LANG,
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=BERTSCORE_RESCALE,
    )
    return float(sum(res["f1"]) / len(res["f1"]))

# https://proceedings.neurips.cc/paper/2021/hash/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Abstract.html?utm_source=chatgpt.com
def compute_bartscore_hypo2ref(preds: List[str], refs: List[str], scorer: BARTScorer) -> float:
    scores = scorer.score(preds, refs, batch_size=BARTSCORE_BATCH)
    return float(sum(scores) / len(scores))  # avg log-likelihood (<=0; less-negative is better)

# https://aclanthology.org/2022.acl-long.229/?utm_source=chatgpt.com
def load_truthfulqa_generation(n: int) -> Tuple[List[str], List[str]]:
    """
    domenicrosati/TruthfulQA (split=train)
    Fields: 'Question', 'Best Answer', 'Correct Answers', ...
    Prompt = Question
    Ref    = Best Answer
    """
    ds = load_dataset("domenicrosati/TruthfulQA", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    prompts = [(r.get("Question") or "").strip() for r in ds]
    refs = [(r.get("Best Answer") or "").strip() for r in ds]
    # filter any empty pairs
    pr, rr = [], []
    for p, r in zip(prompts, refs):
        if p and r:
            pr.append(p); rr.append(r)
    return pr, rr

def get_generation_measures(model_path, prompts, refs, bart_scorer):
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)

    model = model_base.model
    tokenizer = model_base.tokenizer
    device = model.device

    bleu_list, bert_list, bart_list = [], [], []

    for number_of_tokens in number_of_tokens_list:
        print(f"k={number_of_tokens}")
        prompts_pad = tokenize_pad_input_list(prompts, tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
        # convert to device (prompts_pad is a list of dicts with input_ids and attention_mask)
        prompts_pad = [{k: v.to(device) for k, v in prompt.items()} for prompt in prompts_pad]
        completion_list = generate_completion_list(prompts_pad, model, tokenizer)

        # calculate measures
        bleu = compute_bleu(completion_list, refs)
        bert = compute_bertscore_f1(completion_list, refs)
        bart = compute_bartscore_hypo2ref(completion_list, refs, bart_scorer)

        bleu_list.append(bleu)
        bert_list.append(bert)
        bart_list.append(bart)
    return bleu_list, bert_list, bart_list


def main():
    prompts, refs = load_truthfulqa_generation(N_EXAMPLES)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bart_scorer = BARTScorer(device=device, checkpoint=BARTSCORE_CHECKPOINT)

    model_paths = [
        'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-3.1-8B-Instruct',
        'google/gemma-2b-it', 'google/gemma-7b-it',
        'Qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-7B-Chat', 
        'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'
    ]

    # Define CSV file and header
    csv_file = "results/generation_analysis_measures.csv"
    header = ["model", "bleu", "bert", "bart"]

    if not os.path.exists(csv_file):
        # Create CSV file with header
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Iterate and write each model's results
    for model_path in model_paths:
        bleu, bert, bart = get_generation_measures(model_path, prompts, refs, bart_scorer)

        # Append row to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_path, bleu, bert, bart])


if __name__ == "__main__":
    main()