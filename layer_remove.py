import torch
import os
from transformers import LlamaForCausalLM
import subprocess
import json
import argparse

def remove_layers_and_save(model_path, output_dir, layers_to_remove):
    # Load the pre-trained LLaMA model
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove the specified layers
    for layer_idx in layers_to_remove:
        if 0 <= layer_idx < len(model.model.layers):
            del model.model.layers[layer_idx]

    # Renumber the remaining layers' indices
    for layer_idx, module in enumerate(model.model.layers):
        module.self_attn.layer_idx = layer_idx
    
    # Save the modified model
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Update the config.json with the new number of hidden layers
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    config['num_hidden_layers'] = len(model.model.layers)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
    print(f"Updated config saved to {config_path}")


def run_bash_script(script_path, working_directory):
    # Switch to the target directory
    os.chdir(working_directory)

    # Execute the bash script
    subprocess.run(["bash", script_path])

    # Switch back to the original directory
    os.chdir("..")
    print(f"Executed script {script_path} in {working_directory}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Remove layers from LLaMA model and save the modified version.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained LLaMA model")
    parser.add_argument("--layer_index", type=int, required=True, help="Index of the layer to remove")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the modified model")

    args = parser.parse_args()

    # Remove layers and save the model
    remove_layers_and_save(args.model_path, args.save_path, [args.layer_index])

    # Path to the evaluation script and directory
    # target_directory = "/lm-evaluation-harness"
    # script_name = "run_task.sh"

    # Run the evaluation script
    # run_bash_script(script_name, target_directory)
