import torch
import os
import argparse
from transformer_lens import utils
from model_intervention import ModelExperiment

torch.set_grad_enabled(False)

## Basic idea is that we load the model, do interventin (swap or deletion), save the model and run benchmark on it.
## Storage issue - we will save one intervened model for experiment and then delete it.


class ModelEvaluation(ModelExperiment):

    def delete_model_layers_tflens(self, layer):
        """
        Delete a layer from the model.
        """

        def ablate_resid(value, hook):
            return torch.zeros_like(value)

        self.hooked_model.add_hook(utils.get_act_name("attn_out", layer),
                                   ablate_resid)
        self.hooked_model.add_hook(utils.get_act_name("mlp_out", layer),
                                   ablate_resid)

        self.model.load_state_dict(self.hooked_model.state_dict())

    def delete_model_layers(self, layer):
        """
        Delete a layer from the model.
        """
        if 'gpt2' in self.model_name:

            self.model.transformer.h[layer].mlp.c_proj.weight.data.zero_()
            self.model.transformer.h[layer].mlp.c_proj.bias.data.zero_()
            self.model.transformer.h[layer].attn.c_proj.weight.data.zero_()
            self.model.transformer.h[layer].attn.c_proj.bias.data.zero_()
        elif 'pythia' in self.model_name:
            self.model.gpt_neox.layers[layer].attention.dense.weight.zero_()
            self.model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.zero_()
            self.model.gpt_neox.layers[layer].attention.dense.bias.zero_()
            self.model.gpt_neox.layers[layer].mlp.dense_4h_to_h.bias.zero_()


def save_intervened_model(model_name, intervention_type, intervention_layer,
                          save_basedir):
    print("Model loaded")
    intervened_model = ModelEvaluation(
        model_name=model_name, device=f'cuda:{torch.cuda.current_device()}')

    if intervention_type == 'swap':
        intervened_model.swap_model_layers(intervention_layer,
                                           intervention_layer + 1)
    elif intervention_type == 'delete':
        intervened_model.delete_model_layers(intervention_layer)

    print("Model intervention done")

    save_path = os.path.join(save_basedir, intervention_type,
                             f'layer_{str(intervention_layer)}/')
    print(f'Saving model to : {save_path}')
    intervened_model.model.save_pretrained(os.path.join(save_path))
    intervened_model.tokenizer.save_pretrained(os.path.join(save_path))
    print(f"Model saved in {save_path}")

    return save_path


if __name__ == "__main__":
    # Example usage

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--intervention_type', type=str, default="swap")
    parser.add_argument('--intervention_layer', type=int, default=1)
    parser.add_argument('--save_basedir', type=str, default="/")
    args = parser.parse_args()
    save_intervened_model(args.model_name, args.intervention_type,
                          args.intervention_layer, args.save_basedir)
