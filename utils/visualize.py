from PIL import Image

import matplotlib.pyplot as plt


def tensor_to_pil(image):
    """
    Convert PyTorch tensor to PIL Image with normalization.

    Normalizes tensor to [0, 255] range using min-max normalization per image,
    then converts to uint8 PIL Image.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W) with values in any range.

    Returns:
        PIL.Image: PIL Image in RGB format with values in [0, 255].
    """
    image_min = image.amin(dim=(1, 2), keepdim=True)
    image_max = image.amax(dim=(1, 2), keepdim=True)
    cpu_image = ((image - image_min) / (image_max - image_min)).detach().cpu()
    cpu_image = (cpu_image * 255).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(cpu_image)


def plot_grid(images, names, output_path):
    """
    Create and save a grid of images with checkpoint names as labels.

    Generates a matplotlib figure with images arranged in a grid where each row
    corresponds to a checkpoint and each column shows different generated samples.

    Args:
        images (list): List of lists of PIL Images. Outer list = checkpoints,
                      inner list = images per checkpoint.
        names (list): List of checkpoint names (strings) for row labels.
        output_path (str): File path to save the output image.
    """
    num_checkpoints = len(images)
    num_images = len(images[0])

    fig, axes = plt.subplots(num_checkpoints, num_images, figsize=(num_images * 2, num_checkpoints * 2))

    # Normalize axes to 2D array
    if num_checkpoints == 1 and num_images == 1:
        axes = [[axes]]
    elif num_checkpoints == 1:
        axes = [axes]
    elif num_images == 1:
        axes = [[ax] for ax in axes]

    for idx_1, (group_images, name) in enumerate(zip(images, names)):
        for idx_2, img in enumerate(group_images):
            ax = axes[idx_1][idx_2]
            ax.imshow(img)
            ax.axis('off')
            if idx_2 == 0:
                ax.text(-0.1, 0.5, name, transform=ax.transAxes,
                       fontsize=14, va='center', ha='right', fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.show()
