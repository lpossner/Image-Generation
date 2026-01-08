import os
import argparse
import torch

from utils.args import add_visualization_args
from utils.arg_handlers import get_model
from utils.visualize import tensor_to_pil, plot_grid


parser = argparse.ArgumentParser(description="Train model on image dataset")
parser = add_visualization_args(parser)
args = parser.parse_args()

images = []
names = []

for checkpoint_idx, checkpoint_path in enumerate(args.checkpoint_paths):
    model, latent_dim, hidden_dim = get_model(args, checkpoint_idx)
    noise = torch.randn(size=(args.n_images, latent_dim, 1, 1))

    with torch.no_grad():
        image_tensors = model.generate(noise)

    images.append([tensor_to_pil(image_tensor) for image_tensor in image_tensors])
    names.append(os.path.splitext(os.path.basename(checkpoint_path))[0])

plot_grid(images, names, args.output_path)
