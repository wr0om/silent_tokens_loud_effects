from utils import *
from datasets import load_dataset
from AISafetyLab.aisafetylab.evaluation.scorers.llama_guard_3_scorer import LlamaGuard3Scorer

sys.path.append("refusal_direction")
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base


number_of_tokens_list = [0, 1, 2, 4, 8, 16, 32, 64, 128]
results_path = "results/safety_analysis_measures.csv"


def main():
    dataset = load_dataset("thu-coai/AISafetyLab_Datasets", "harmbench", split="standard")
    prompts = [d['query'] for d in dataset]

    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)

    model = model_base.model
    tokenizer = model_base.tokenizer
    device = model.device

    number_of_tokens_2_completions = {}
    for number_of_tokens in tqdm.tqdm(number_of_tokens_list):
        prompts_pad = tokenize_pad_input_list(prompts, tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
        prompts_pad = [{k: v.to(device) for k, v in prompt.items()} for prompt in prompts_pad]
        completions = generate_completion_list(prompts_pad, model, tokenizer)
        number_of_tokens_2_completions[number_of_tokens] = completions
    
    del model_base

    guard_model_path = 'meta-llama/Llama-Guard-3-8B'
    llama_guard_scorer = LlamaGuard3Scorer(model_path=guard_model_path)
    mean_scores_list, std_score_list = [], []
    for number_of_tokens, completions in tqdm.tqdm(number_of_tokens_2_completions.items()):
        scores = [llama_guard_scorer.score(query, response)['score'] \
            for query, response in zip(prompts, completions)]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_scores_list.append(mean_score)
        std_score_list.append(std_score)
        print(f"number_of_tokens: {number_of_tokens}, mean_score: {mean_score}, std_score: {std_score}")

    # Save results to CSV
    csv_file = "results/safety_analysis_measures.csv"
    header = ["model", "llama_guard3_scores", "std_llama_guard3_scores"]

    if not os.path.exists(csv_file):
        # Create CSV file with header
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append row to CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_alias, mean_scores_list, std_score_list])


if __name__ == "__main__":
    main()