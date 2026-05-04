import argparse
import os
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MinecraftDataset
from diffusion import compute_loss
from model import UNetTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--min_fill", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--transformer_layers", type=int, default=8)
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--air_weight", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=500000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()


def warmup_schedule(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MinecraftDataset(
        data_dirs=args.data_dirs,
        chunk_size=args.chunk_size,
        min_fill=args.min_fill,
        cache_dir=args.cache_dir,
    )

    n_val = max(1, int(args.val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = UNetTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        time_dim=args.time_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_schedule(s, args.warmup_steps)
    )

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    run_dir = os.path.join(
        args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(
            {
                "block_to_idx": dataset.block_to_idx,
                "idx_to_block": dataset.idx_to_block,
                "vocab_size": dataset.vocab_size,
                "mask_idx": dataset.mask_idx,
            },
            f,
        )

    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        print(f"Resumed from step {global_step}")

    def save_checkpoint(step: int, epoch: int):
        path = os.path.join(run_dir, f"ckpt_step{step:07d}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "epoch": epoch,
                "vocab_size": dataset.vocab_size,
                "args": vars(args),
            },
            path,
        )
        print(f"\nSaved checkpoint: {path}")

    model.train()
    epoch = start_epoch

    while global_step < args.max_steps:
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            blocks = batch["blocks"].to(device)
            condition_mask = batch["condition_mask"].to(device)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = compute_loss(model, blocks, condition_mask, args.air_weight)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss = compute_loss(model, blocks, condition_mask, args.air_weight)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

            sched.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if global_step % 100 == 0:
                lr = sched.get_last_lr()[0]
                tqdm.write(
                    f"step={global_step:7d}  loss={loss_val:.4f}  lr={lr:.2e}"
                )

            if global_step % args.val_every == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        vb = val_batch["blocks"].to(device)
                        vc = val_batch["condition_mask"].to(device)
                        vl = compute_loss(model, vb, vc, args.air_weight)
                        val_losses.append(vl.item())
                avg_val = sum(val_losses) / max(1, len(val_losses))
                tqdm.write(f"  val_loss={avg_val:.4f}")
                model.train()

            if global_step % args.save_every == 0:
                save_checkpoint(global_step, epoch)

            if global_step >= args.max_steps:
                break

        avg_train = epoch_loss / max(1, epoch_steps)
        print(f"Epoch {epoch} done — avg train loss: {avg_train:.4f}")
        epoch += 1

    save_checkpoint(global_step, epoch)
    print("Training complete.")


if __name__ == "__main__":
    main()
