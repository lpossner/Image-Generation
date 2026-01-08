import os

import torch


def load_checkpoint(checkpoint_directory, trainer, resume_from=None):
    """
    Load checkpoint to resume training from saved state.

    Searches for the most recent checkpoint or a specific checkpoint if
    resume_from is specified. Loads model weights, optimizer state, and metrics.

    Args:
        checkpoint_directory (str): Directory containing checkpoint files.
        trainer: Trainer object with model, optimizer, and metrics attributes.
        resume_from (int, optional): Specific epoch number to resume from.
                                     Use -1 to force starting from scratch.
                                     If None, loads the most recent checkpoint.

    Returns:
        tuple: (trainer, start_epoch) where:
            - trainer: Updated trainer with loaded checkpoint state.
            - start_epoch (int): Epoch number to start/resume training from.
    """
    if resume_from == -1:
        print("User forced starting from scratch")
        start_epoch = 0
        return trainer, start_epoch
    
    checkpoints = [
        filename
        for filename in os.listdir(checkpoint_directory)
        if trainer.checkpoint_name in filename and filename.endswith(".pth")
    ]
    if not checkpoints:
        print("No checkpoints found. Starting from scratch")
        start_epoch = 0
        return trainer, start_epoch
    
    load_checkpoint = max(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if resume_from:
        load_checkpoint = f"{trainer.checkpoint_name}_{resume_from}.pth"
    checkpoint_path = os.path.join(checkpoint_directory, load_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.metrics = checkpoint["metrics"]
    start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
    print(f"Resuming training from epoch {start_epoch}")
    return trainer, start_epoch


def safe_checkpoint(checkpoint_directory, trainer, epoch, metrics):
    """
    Save training checkpoint to disk.

    Saves model state, optimizer state, current epoch, and metrics to a
    checkpoint file for later resumption of training.

    Args:
        checkpoint_directory (str): Directory to save checkpoint file.
        trainer: Trainer object containing model, optimizer, and checkpoint_name.
        epoch (int): Current epoch number (0-indexed).
        metrics (dict): Dictionary of training metrics to save.
    """
    checkpoint_path = os.path.join(checkpoint_directory, f"{trainer.checkpoint_name}_{epoch + 1}.pth")
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")
