from utils.logger import Logger
from utils.device import detect_device

def get_data_loader(args):
    if args.dataset == "CelebA":
        from datasets.CelebA import CelebADatalLoader as DataLoader
    else:
        raise NotImplementedError("Dataset {args.dataset} is not implemented")
    
    return DataLoader(args.img_directory, args.batch_size)


def get_trainer(args):
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
