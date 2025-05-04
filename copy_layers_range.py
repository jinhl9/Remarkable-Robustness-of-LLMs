from typing import List

import os
import argparse
import joblib as jl
import time
import torch
import copy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_intervention import ModelExperiment, _extract_layer_prefixes

torch.set_grad_enabled(False)

import os

os.environ['HF_HOME'] = '/ceph/scratch/jlee/.cache/huggingface'
os.environ[
    'HF_DATASETS_CACHE'] = '/ceph/scratch/jlee/.cache/huggingface/datasets'


class CopyModel(ModelExperiment):

    def __init__(self, model_name="openai-community/gpt2", device='cuda:1'):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        if 'gpt2' in model_name:
            self.n_layer = self.model.config.n_layer
        elif 'pythia' in model_name or 'Qwen' in model_name or 'llama' in model_name:
            self.n_layer = self.model.config.num_hidden_layers

    def copy_layers(self, copy_start: int, copy_end: int, num_copy: int = 1):
        """
        Copy layers of the model from copy_indices to insertion_indices.
        Args:
            copy_indices (List[int]): Indices of layers to copy from.
            insertion_indices (List[int]): Indices of layers to copy to.
        """

        self.model.to(device)
        self.copied_model = copy.deepcopy(self.model)

        model_prefix = _extract_layer_prefixes(self.model)
        model_prefix_list = model_prefix.split('.')
        original_model_blocks = getattr(
            getattr(self.model, model_prefix_list[0]), model_prefix_list[1])
        new_blocks = []
        new_blocks_idx = []
        for copy_index in range(copy_start, copy_end):
            ## Copy entire block
            copied_layer = copy.deepcopy(original_model_blocks[copy_index])
            new_blocks += [copied_layer] * (num_copy + 1)
            new_blocks_idx += [copy_index] * (num_copy + 1)
            if 'gpt2' in model.model_name:
                self.copied_model.config.n_layer += num_copy
            elif 'pythia' in model.model_name or 'Qwen' in model.model_name or 'llama' in model.model_name:
                self.copied_model.config.num_hidden_layers += num_copy
        #print('new_block: ', new_blocks_idx)
        new_model_blocks = original_model_blocks[:
                                                 copy_start] + new_blocks + original_model_blocks[
                                                     copy_end:]
        #print(len(new_model_blocks), len(original_model_blocks))
        setattr(getattr(self.copied_model, model_prefix_list[0]),
                model_prefix_list[1], new_model_blocks)


def calculate_base2_entropy(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-9)
    log_probs = torch.log(probs) / torch.log(torch.tensor(2.0))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name",
        type=str,
        default="openai-community/gpt2",
        help="Name of the model to use.",
    )
    argparser.add_argument(
        "--copy_start",
        type=int,
        default=10,
        help="Indices of layers to copy from.",
    )

    argparser.add_argument(
        "--num_copy",
        type=int,
        default=1,
        help="number of copys to make.",
    )

    argparser.add_argument(
        "--save_dir",
        type=str,
        default="results/copy_layers_range",
        help="Directory to save the results.",
    )

    # Load the model and tokenizer
    args = argparser.parse_args()

    device = f'cuda:{torch.cuda.current_device()}'

    for _ in range(5):
        try:
            dataset = load_dataset("EleutherAI/the_pile_deduplicated",
                                   split='train',
                                   streaming=True)
            break
        except Exception as e:
            print(f"Retrying in 10s: {e}")
            time.sleep(10)

    model = CopyModel(args.model_name, device=device)
    print("Model loaded")
    datapoints = 10
    data = dataset.take(datapoints)

    for copy_last in range(args.copy_start, model.n_layer + 1):
        save_dir = os.path.join(
            args.save_dir,
            args.model_name,
            'num_copy_' + str(args.num_copy),
            str(args.copy_start),
            str(copy_last),
        )
        if os.path.isfile(os.path.join(save_dir, 'entropy_all.jl')):
            continue

        print('Save to: ', save_dir)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model.copy_layers(args.copy_start, copy_last, args.num_copy)
        print("Layers copied")
        entropy_all = []
        for idx, sample in enumerate(data):
            inputs = model.tokenizer(sample["text"],
                                     return_tensors="pt",
                                     truncation=True,
                                     max_length=512).to(device)
            with torch.no_grad():
                logits = model.copied_model(**inputs).logits
                entropies = calculate_base2_entropy(logits)
            entropy_all.append(entropies)
        jl.dump({'entropy': entropy_all},
                os.path.join(save_dir, 'entropy_all.jl'))
