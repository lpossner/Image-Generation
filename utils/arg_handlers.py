from utils.logger import Logger
from utils.device import detect_device

def get_data_loader(args):
    """
    Create and return appropriate data loader based on dataset argument.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
                                   dataset name, image_directory, and batch_size.

    Returns:
        DataLoader: Configured data loader for the specified dataset.

    Raises:
        NotImplementedError: If the specified dataset is not implemented.
    """
    if args.dataset == "CelebA":
        from datasets.CelebA import CelebADatalLoader as DataLoader
    else:
        raise NotImplementedError("Dataset {args.dataset} is not implemented")
    
    return DataLoader(args.image_directory, args.batch_size)


def get_trainer(args):
    """
    Create and return appropriate trainer based on model argument.

    Initializes model, data loader, logger, and trainer with appropriate
    hyperparameters for the specified model type (VAE, DCGAN, or WGAN).

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all
                                   necessary configuration parameters.

    Returns:
        Trainer: Configured trainer object (VAETrainer, DCGANTrainer, or WGANTrainer).

    Raises:
        NotImplementedError: If the specified model is not implemented.
    """
    checkpoint_name = f"{args.model}_{args.dataset}"
    data_loader = get_data_loader(args)
    logger = Logger(checkpoint_name, args.log_directory, args.neptune_project, args.neptune_token)
    device = detect_device(args.device)
    if args.model == "VAE":
        from models.VAE import VAE
        from trainers.VAE import VAETrainer
        model = VAE(num_colors=3, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
        trainer = VAETrainer(model, device, data_loader, checkpoint_name, logger, args.lr)
    elif args.model == "DCGAN":
        from models.DCGAN import DCGAN
        from trainers.DCGAN import DCGANTrainer
        model = DCGAN(num_colors=3, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
        trainer = DCGANTrainer(model, device, data_loader, checkpoint_name, logger, args.lr_generator, args.lr_discriminator, args.beta1, args.beta2, args.latent_dim)
    elif args.model == "WGAN":
        from models.WGAN import WGAN
        from trainers.WGAN import WGANTrainer
        model = WGAN(num_colors=3, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
        trainer = WGANTrainer(model, device, data_loader, checkpoint_name, logger, args.lr_generator, args.lr_discriminator, args.beta1, args.beta2, args.latent_dim, args.lambda_gp, args.n_critic)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    
    return trainer


def get_model(args, checkpoint_idx):
    """
    Load model from checkpoint with automatically inferred dimensions.

    Reads checkpoint file and infers latent_dim and hidden_dim from saved
    weight shapes to reconstruct the model architecture.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing model
                                   type and checkpoint_paths.
        checkpoint_idx (int): Index of the checkpoint to load from checkpoint_paths list.

    Returns:
        tuple: (model, latent_dim, hidden_dim) where:
            - model: Loaded model with weights from checkpoint.
            - latent_dim (int): Inferred latent dimension.
            - hidden_dim (int): Inferred hidden dimension.

    Raises:
        NotImplementedError: If the specified model is not implemented.
    """
    import torch
    checkpoint = torch.load(args.checkpoint_paths[checkpoint_idx], map_location=torch.device('cpu'))
    if args.model == "VAE":
        from models.VAE import VAE
        fc_mu_weights = checkpoint['model_state_dict']['encoder.fc_mu.weight']
        encoder_weights = checkpoint['model_state_dict']['encoder.encoder.0.weight']
        latent_dim = fc_mu_weights.size(1)  # Output dimenion of mu layer
        hidden_dim = encoder_weights.size(1)  # Output dimenion of first encoder layer
        model = VAE(num_colors=3, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model == "DCGAN":
        from models.DCGAN import DCGAN
        weights = checkpoint['model_state_dict']['generator.generator.0.weight']
        latent_dim = weights.size(0)  # Input dimenion of first generator layer
        hidden_dim = int(weights.size(1) / 8)  # Output dimension of first generator layer
        model = DCGAN(num_colors=3, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model == "WGAN":
        from models.WGAN import WGAN
        weights = checkpoint['model_state_dict']['generator.generator.0.weight']
        latent_dim = weights.size(0)  # Input dimenion of first generator layer
        hidden_dim = int(weights.size(1) / 16)  # Output dimension of first generator layer
        model = WGAN(num_colors=3, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    
    return model, latent_dim, hidden_dim
