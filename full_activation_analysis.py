from utils import *


def reduce_activations(activation_groups, n_components=2, method="tsne", losses_groups=None, seed=42,**kwargs):
    """
    Reduce dimensionality of activations using t-SNE, or PCA, handling an unlimited number of activation groups.

    Args:
        activation_groups (list): List of lists of NumPy arrays. Each sublist represents a group of activations.
        method (str): Dimensionality reduction method ('tsne', 'umap', or 'pca').
        losses_groups (list): List of lists of loss values corresponding to each activation group.
        kwargs: Additional parameters for t-SNE or PCA.

    Returns:
        list: List of 2D or 3D reduced activations corresponding to each input group, optionally including loss.
    """
    # Flatten all activations from all groups into a single array
    combined_activations = np.vstack([np.vstack(group) for group in activation_groups])

    # Reduce dimensionality
    if method.lower() == "tsne":
        # Set perplexity dynamically based on the number of samples
        n_samples = combined_activations.shape[0]
        perplexity = min(kwargs.get("perplexity", 30), n_samples - 1)  # Default to 30 if not specified
        min_grad_norm = kwargs.get("min_grad_norm", 1e-7)  # Default t-SNE min_grad_norm if not specified

        tsne = TSNE(n_components=n_components, perplexity=perplexity, min_grad_norm=min_grad_norm, random_state=seed)
        reduced_activations = tsne.fit_transform(combined_activations)
    elif method.lower() == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        reduced_activations = pca.fit_transform(combined_activations)
    else:
        raise ValueError("Method must be 'tsne', 'umap', or 'pca'.")

    # Split reduced activations back into original groups
    reduced_groups = []
    start_idx = 0
    for group in activation_groups:
        group_size = len(np.vstack(group))
        reduced_groups.append(reduced_activations[start_idx:start_idx + group_size])
        start_idx += group_size

    # Add loss to final dimensionality-reduced activations if provided
    if losses_groups:
        combined_losses = np.hstack([np.hstack(group) for group in losses_groups])
        if len(combined_losses) != len(reduced_activations):
            raise ValueError("Losses must correspond to the number of activations.")

        # Append loss as an additional dimension to the reduced activations
        reduced_activations_with_loss = np.hstack([reduced_activations, combined_losses[:, np.newaxis]])

        # Split reduced activations with loss back into original groups
        reduced_groups_with_loss = []
        start_idx = 0
        for group in activation_groups:
            group_size = len(np.vstack(group))
            reduced_groups_with_loss.append(reduced_activations_with_loss[start_idx:start_idx + group_size])
            start_idx += group_size

        return reduced_groups_with_loss

    return reduced_groups


number_of_tokens_list = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def get_activation_measures(model_path, plot_clusters=False):
	model_alias = os.path.basename(model_path)
	cfg = Config(model_alias=model_alias, model_path=model_path)
	model_base = construct_model_base(cfg.model_path)
	harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)


	## activation similarity
	# get activations of harmless_train
	harmless_activations = get_activations_array(model_base.model, harmless_train,\
												model_base.tokenize_instructions_fn, \
													model_base.model_block_modules, batch_size=1, positions=[-1]).squeeze()
	# compute similarity over different padding sizes
	deltas = []
	for number_of_tokens in number_of_tokens_list:
		harmless_train_pad = tokenize_pad_input_list(harmless_train, model_base.tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
		harmless_activations_pad = tokenized_get_activations_array(model_base.model, harmless_train_pad, model_base.model_block_modules).squeeze()

		per_prompt_similarities = []
		for i in range(len(harmless_train)):
			per_prompt_similarity = torch.nn.functional.cosine_similarity(harmless_activations[i], harmless_activations_pad[i], dim=-1).squeeze().cpu().numpy()
			per_prompt_similarities.append(per_prompt_similarity)

		per_prompt_similarities = np.array(per_prompt_similarities)
		mean_prompt_similarities = np.mean(per_prompt_similarities, axis=1)
		delta = np.mean(mean_prompt_similarities)
		deltas.append(delta)

	## Refusal Similarity
	harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
	harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)

	refusal_direction = generate_directions(
			model_base,
			harmful_train,
			harmless_train,
			artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))
	refusal_direction = refusal_direction[-1, :, :].unsqueeze(0) # 1, n_layer, d_model

	# compute similarity over different padding sizes
	R_scores = []
	for number_of_tokens in number_of_tokens_list:
		harmless_train_pad = tokenize_pad_input_list(harmless_train, model_base.tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
		harmful_train_pad = tokenize_pad_input_list(harmful_train, model_base.tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)

		refusal_direction_pad = tokenized_generate_directions(
			model_base,
			harmful_train_pad,
			harmless_train_pad,
			artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions_tokenized")
		)
		refusal_direction_pad = refusal_direction_pad[-1, :, :].unsqueeze(0) # 1, n_layer, d_model
		cosine_similarity = torch.nn.functional.cosine_similarity(refusal_direction, refusal_direction_pad, dim=-1).squeeze().cpu().numpy()
		R_score = np.mean(cosine_similarity)
		R_scores.append(R_score)

	## Clustering
	# compute activations with different padding sizes
	silhouette_scores = []
	for number_of_tokens in number_of_tokens_list:
		harmless_train_pad = tokenize_pad_input_list(harmless_train, model_base.tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
		harmful_train_pad = tokenize_pad_input_list(harmful_train, model_base.tokenizer, number_of_tokens, model_base.tokenize_instructions_fn)
		# taking last layer representations
		harmless_activations_pad = tokenized_get_activations_array(model_base.model, harmless_train_pad, model_base.model_block_modules).squeeze()[:,-1,:]
		harmful_activations_pad = tokenized_get_activations_array(model_base.model, harmful_train_pad, model_base.model_block_modules).squeeze()[:,-1,:]
		harmless_activations_pad = harmless_activations_pad.cpu().numpy()
		harmful_activations_pad = harmful_activations_pad.cpu().numpy()

		# Combine embeddings and labels
		X = np.vstack([harmless_activations_pad, harmful_activations_pad])
		y = np.array([0]*len(harmless_activations_pad) + [1]*len(harmful_activations_pad))

		score = silhouette_score(X, y)
		silhouette_scores.append(score)

		if plot_clusters:
			# reduce to 2D using PCA
			harmless_2D, harmful_2D = \
				reduce_activations([harmless_activations_pad, harmful_activations_pad], n_components=2, method='pca')

			# plot 2D activations of classes
			plt.figure(figsize=(8, 6))
			plt.scatter(harmless_2D[:, 0], harmless_2D[:, 1], label='Harmless', alpha=0.7, color='blue', s=20)
			plt.scatter(harmful_2D[:, 0], harmful_2D[:, 1], label='Harmful', alpha=0.7, color='red', s=20)
			plt.title(f"2D PCA of Activations (Padding Size: {number_of_tokens})")
			plt.xlabel("Component 1")
			plt.ylabel("Component 2")
			plt.legend()
			plt.grid()
			plt.show()

	return deltas, R_scores, silhouette_scores


def main():
    model_paths = [
        'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-3.1-8B-Instruct',
        'google/gemma-2b-it', 'google/gemma-7b-it',
        'Qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-7B-Chat', 'Qwen/Qwen-14B-Chat',
        'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'
    ]

    # Define CSV file and header
    csv_file = "results/activation_analysis_measures.csv"
    header = ["model", "delta_scores", "R_scores", "silhouette_scores"]

    # Create CSV file with header
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Iterate and write each model's results
    for model_path in model_paths:
        deltas, R_scores, silhouette_scores = get_activation_measures(model_path)

        # Append row to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_path, deltas, R_scores, silhouette_scores])


if __name__ == "__main__":
	main()