import torch
from torch.utils.data import DataLoader
from improved_diffusion.script_util import create_gaussian_diffusion
from minecraft_dataset import MinecraftSchematicDataset
from improved_diffusion.unet import UNetModel
from improved_diffusion.train_util import TrainLoop
import os
import argparse
from improved_diffusion import dist_util, logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--ema_rate", type=str, default="0.9999")
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_anneal_steps", type=int, default=0)
    return parser.parse_args()

def train(
    model,
    diffusion,
    train_dataloader,
    batch_size,
    learning_rate,
    save_interval,
    log_interval,
    resume_checkpoint,
    use_fp16,
    ema_rate,
    microbatch,
    weight_decay,
    lr_anneal_steps,
):
    """
    Train the diffusion model.
    """
    logger.configure(dir="logs", format_strs=["stdout", "log", "csv"])
    
    # Create a data iterator that yields batches of blocks and masks
    def data_iterator():
        while True:
            for batch in train_dataloader:
                # Convert blocks to one-hot encoding
                blocks = batch["blocks"]
                mask = batch["mask"]
                
                # Create a batch of data for the diffusion model
                # The model expects a batch of shape [batch_size, channels, height, width, depth]
                # where channels is the number of block types (one-hot encoded)
                batch_size = blocks.shape[0]
                num_blocks = model.in_channels
                
                # Create one-hot encoding
                one_hot = torch.zeros(
                    (batch_size, num_blocks, 16, 16, 16),
                    device=dist_util.dev()
                )
                
                # Fill in the one-hot encoding
                for b in range(batch_size):
                    for y in range(16):
                        for z in range(16):
                            for x in range(16):
                                block_idx = blocks[b, y, z, x].item()
                                one_hot[b, block_idx, y, z, x] = 1.0
                
                # Create condition dictionary with mask
                cond = {"mask": mask.to(dist_util.dev())}
                
                yield one_hot.to(dist_util.dev()), cond
    
    # Create the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_iterator(),
        batch_size=batch_size,
        microbatch=microbatch,
        lr=learning_rate,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
    ).run_loop()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create the dataset
    dataset = MinecraftSchematicDataset(
        schematics_dir="minecraft-schematics-raw",
        chunk_size=16,
        cache_file="cache/block_mappings.pkl",
        max_files=None,  # Use all files
    )

    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # Create the model
    model = UNetModel(
        in_channels=len(dataset.block_to_idx),
        model_channels=64,
        out_channels=len(dataset.block_to_idx),
        num_res_blocks=2,
        attention_resolutions=(4,),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
    )
    
    # Move model to device
    model.to(dist_util.dev())
    
    # Create the diffusion process
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        sigma_small=False,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    
    # Train the model
    train(
        model=model,
        diffusion=diffusion,
        train_dataloader=train_dataloader,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        ema_rate=args.ema_rate,
        microbatch=args.microbatch,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )
