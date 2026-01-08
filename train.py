import argparse

from utils.args import add_general_args, add_training_args, add_neptune_args
from utils.checkpoint import load_checkpoint, safe_checkpoint


from utils.arg_handlers import get_trainer


parser = argparse.ArgumentParser(description="Train model on image dataset")
parser = add_general_args(parser)
parser = add_training_args(parser)
parser = add_neptune_args(parser)
args = parser.parse_args()

trainer = get_trainer(args)

trainer, start_epoch = load_checkpoint(
    args.checkpoint_directory, trainer, args.resume_from
)

for epoch in range(start_epoch, args.epochs):
    metrics = trainer.train_epoch(epoch, args.epochs)
    safe_checkpoint(
        args.checkpoint_directory,
        trainer,
        epoch,
        metrics,
    )
