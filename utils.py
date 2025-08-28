import torch
import random
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import csv
import tqdm
import math
from typing import List, Dict, Tuple


# go back .. directory
import sys
sys.path.append("refusal_direction")

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions, tokenized_generate_directions, get_activations_array, tokenized_get_activations_array
from pipeline.run_pipeline import load_and_sample_datasets



def tokenize_pad_input(prompt, tokenizer, number_of_tokens, tokenize_instructions_fn, good_attention_mask=False):
	"Tokenize the input prompt, and then pad it (left padding) number_of_tokens times using the pad token."
	pad_token_id = tokenizer.pad_token_id
	# Tokenize the input prompt
	tokenized_input = tokenize_instructions_fn(instructions=[prompt])
	# Pad the input and
	padded_input = tokenized_input["input_ids"]
	for _ in range(number_of_tokens):
		padded_input = torch.cat([torch.tensor([[pad_token_id]]), padded_input], dim=1)

	# get mask for padded input
	if good_attention_mask:
		attention_mask = torch.ones_like(padded_input)
		attention_mask[padded_input == pad_token_id] = 0
	else:
		attention_mask = torch.ones_like(padded_input)

	# put in a dict with input_ids and attention_mask
	padded_input = {
		"input_ids": padded_input,
		"attention_mask": attention_mask
	}
	return padded_input


def tokenize_pad_input_list(prompts, tokenizer, number_of_tokens, tokenize_instructions_fn, good_attention_mask=False):
	"Tokenize the input prompts, and then pad them (left padding) number_of_tokens times using the pad token."
	padded_inputs = []
	for prompt in prompts:
		padded_input = tokenize_pad_input(prompt, tokenizer, number_of_tokens, tokenize_instructions_fn, good_attention_mask)
		padded_inputs.append(padded_input)
	return padded_inputs

def generate_completion(tokenized_input, model, tokenizer):
	with torch.no_grad():
		output = model.generate(
			input_ids=tokenized_input["input_ids"],
			attention_mask=tokenized_input.get("attention_mask"),
			max_new_tokens=64,
			use_cache=False,  # <-- minimal fix for Qwen's past_key_values None bug
		)
	output = output[:, tokenized_input['input_ids'].shape[-1]:]  # take only the generation
	decoded_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
	return decoded_output

def generate_completion_list(tokenized_inputs, model, tokenizer):
	completion_list = []
	for tokenized_input in tqdm.tqdm(tokenized_inputs):
		completion = generate_completion(tokenized_input, model, tokenizer)
		completion_list.append(completion)
	return completion_list