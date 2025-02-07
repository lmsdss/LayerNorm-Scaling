import argparse
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from short_hf import ShortHFModel

def compute_angular_distance(model, data, max_seq_len=1024, stride=256, n_samples=50):
    angular_distances = []
    for i, batch in enumerate(tqdm(data)):
        if i >= n_samples:
            break
        prompts = batch['text']

        model.eval_importance(
            prompts=prompts,
            max_seq_len=max_seq_len,
            stride=stride,
            max_gen_len=0,
            angular=True
        )
    angular_distances.extend(model.importances)
    return angular_distances

def plot_angular_distances(distances, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, label='Angular Distances')
    plt.xlabel('Sample Index')
    plt.ylabel('Angular Distance')
    plt.title('Angular Distances Across Samples')
    plt.legend()
    plt.savefig(output_path)
    plt.show()

def main(args):
    model = ShortHFModel(
        model_name=args.model_path,
        layers_path="model.layers",
        n_prune_layers=1, # this is a dummy value, don't worry about it
    )

    data = load_dataset("allenai/c4", "en", split="train", streaming=True)

    angular_distances = compute_angular_distance(model, data, args.max_seq_len, args.stride, args.n_samples)

    plot_angular_distances(angular_distances, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot angular distances")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to process")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for processing sequences")
    parser.add_argument("--output_path", type=str, default="angular_distances.png", help="Output path for the plot")

    args = parser.parse_args()

    main(args)