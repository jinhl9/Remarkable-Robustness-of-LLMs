"""
BAsed on  https://github.com/AntixK/PyTorch-Model-Compare
"""
import argparse
import os

import torch

from datasets import load_dataset

from model_intervention import ModelExperiment

# Load model and run it on a batch

torch.set_grad_enabled(False)


def _HSIC(K, L, device):
    """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
    N = K.shape[0]
    ones = torch.ones(N, 1).to(device)
    result = torch.trace(K @ L)
    result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) *
                                                              (N - 2))).item()
    result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
    return (1 / (N * (N - 3)) * result).item()


class ModelActivation(ModelExperiment):

    def get_activation(self, tokens):
        _, cache = self.hooked_model.run_with_cache(tokens)
        layer_acts = [
            cache["resid_post", i].reshape(-1, self.hooked_model.cfg.d_model)
            for i in range(self.hooked_model.cfg.n_layers)
        ]
        return layer_acts


def CKA_layers(hsic_matrix, activations, device):

    for i, (layer1_feature) in enumerate(activations):
        X = layer1_feature.flatten(1)
        K = X @ X.t()
        K.fill_diagonal_(0.0)
        hsic_matrix[i, :, 0] += _HSIC(K, K, device)

        for j, (layer2_feature) in enumerate(activations):
            Y = layer2_feature.flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0)
            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            hsic_matrix[i, j, 1] += _HSIC(K, L, device)
            hsic_matrix[i, j, 2] += _HSIC(L, L, device)

    return hsic_matrix


def get_cka_layers(model_name, num_samples):
    torch.cuda.empty_cache()
    device = f'cuda:{torch.cuda.current_device()}'

    model = ModelActivation(model_name=model_name, device=device)

    dataset = load_dataset("EleutherAI/the_pile_deduplicated",
                           split='train',
                           streaming=True)
    hsic_matrix = torch.zeros(model.hooked_model.cfg.n_layers,
                              model.hooked_model.cfg.n_layers, 3)
    batch_count = 0
    for sample in dataset.take(num_samples):
        tokens = model.hooked_model.to_tokens(sample['text'])
        activations = model.get_activation(tokens)
        batch_count += 1
        #print(activations[0].shape)  # len tokens, d_model
        hsic_matrix = CKA_layers(hsic_matrix, activations, device)

    hsic_matrix /= batch_count
    hsic_matrix = hsic_matrix[:, :, 1] / (hsic_matrix[:, :, 0].sqrt() *
                                          hsic_matrix[:, :, 2].sqrt())
    assert not torch.isnan(
        hsic_matrix).any(), "HSIC computation resulted in NANs"
    return hsic_matrix, batch_count


if __name__ == '__main__':
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='openai-community/gpt2')
    parser.add_argument('--num_samples', type=int, default=1388)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/ceph/scratch/jlee/llm-robustness/repsim-layers-cka')
    args = parser.parse_args()

    save_path = os.path.join(args.save_dir, f"{args.model_name}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('Start computing CKA')
    hsic_matrix, batch_count = get_cka_layers(args.model_name,
                                              args.num_samples)
    print(save_path + f"/{args.model_name.split('/')[-1]}.pt")
    torch.save({
        'hsic_matrix': hsic_matrix,
        'batch_count': batch_count,
    }, save_path + f"/{args.model_name.split('/')[-1]}.pt")
