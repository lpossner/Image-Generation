def add_general_args(parser):
    """
    Add general command-line arguments to argument parser.

    Adds arguments for model selection, dataset, training configuration,
    checkpoints, and device selection.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: Parser with general arguments added.
    """
    parser.add_argument(
        "--model", type=str, required=True, help="Name of model (VAE, DCGAN or WGAN)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of dataset (CelebA)"
    )
    parser.add_argument(
        "--image-directory", type=str, required=True, help="Path to image directory"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--checkpoint-directory",
        type=str,
        default=".checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument("--device", type=str, help="Device (cpu, mps or cuda)")
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Number of checkpoint to resume from",
    )
    parser.add_argument(
        "--log-directory",
        type=str,
        default=".logs",
        help="Directory for logs",
    )
    return parser


def add_training_args(parser):
    """
    Add training-specific command-line arguments to argument parser.

    Adds arguments for model dimensions, learning rates, optimizer parameters,
    and WGAN-specific settings.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: Parser with training arguments added.
    """
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (only for VAE)"
    )
    parser.add_argument(
        "--lr_generator",
        type=float,
        default=1e-4,
        help="Generator learning rate (only for DCGAN and WGAN)",
    )
    parser.add_argument(
        "--lr_discriminator",
        type=float,
        default=1e-4,
        help="Discriminator learning rate (only for DCGAN)",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=2e-4,
        help="Critic learning rate (only for WGAN)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta 1 (only for DCGAN and WGAN)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta 2 (only for DCGAN and WGAN)"
    )
    parser.add_argument(
        "--lambda-gp",
        type=float,
        default=0.999,
        help="Lambda for gradient pentalty (only for WGAN)",
    )
    parser.add_argument(
        "--n-critic",
        type=int,
        default=5,
        help="Number of critic updates before generator gets updated (only for WGAN)",
    )
    return parser


def add_neptune_args(parser):
    """
    Add Neptune logging arguments to argument parser.

    Adds arguments for Neptune.ai cloud-based experiment tracking.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: Parser with Neptune arguments added.
    """
    parser.add_argument(
        "--neptune-project", type=str, default="", help="Neptune project name"
    )
    parser.add_argument(
        "--neptune-token", type=str, default="", help="Neptune API token"
    )
    return parser


def add_visualization_args(parser):
    """
    Add visualization-specific command-line arguments to argument parser.

    Adds arguments for loading model checkpoints and generating image grids.

    Args:
        parser (argparse.ArgumentParser): Argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: Parser with visualization arguments added.
    """
    parser.add_argument(
        "--model", type=str, required=True, help="Name of model (VAE, DCGAN or WGAN)"
    )
    parser.add_argument(
        "--checkpoint-paths",
        required=True,
        nargs="+",
        default=[],
        help="Paths of checkpoints",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=str,
        help="Output Path of plots",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of images to visualize",
    )
    return parser
