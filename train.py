import argparse
import copy
import math
import os
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MinecraftDataset
from diffusion import compute_loss, compute_accuracy
from model import UNetTransformer
from viz import save_sample_viz, save_3d_viz


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
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=500000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()


def lr_schedule(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
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
        chunk_size=args.chunk_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: lr_schedule(s, args.warmup_steps, args.max_steps)
    )

    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

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
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"New weights (randomly initialized): {missing}")
        if unexpected:
            print(f"Dropped weights (not in model): {unexpected}")
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"], strict=False)
        global_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        try:
            optim.load_state_dict(ckpt["optim"])
            sched.load_state_dict(ckpt["sched"])
        except (ValueError, RuntimeError):
            print(
                "Optimizer state incompatible — fresh optimizer, scheduler fast-forwarded"
            )
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=lambda s: lr_schedule(s, args.warmup_steps, args.max_steps),
                last_epoch=global_step - 1,
            )
        print(f"Resumed from step {global_step}")

    def save_checkpoint(step: int, epoch: int):
        path = os.path.join(run_dir, f"ckpt_step{step:07d}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
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

    _viz_loader = DataLoader(val_set, batch_size=10, shuffle=False, num_workers=0)
    _viz_batch = next(iter(_viz_loader))
    fixed_viz_blocks = _viz_batch["blocks"].to(device)
    fixed_viz_mask = _viz_batch["condition_mask"].to(device)

    model.train()
    epoch = start_epoch

    while global_step < args.max_steps:
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            blocks = batch["blocks"].to(device)
            condition_mask = batch["condition_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = compute_loss(model, blocks, condition_mask, args.air_weight, valid_mask)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss = compute_loss(model, blocks, condition_mask, args.air_weight, valid_mask)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

            sched.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1.0 - args.ema_decay)

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if global_step % 100 == 0:
                lr = sched.get_last_lr()[0]
                tqdm.write(f"step={global_step:7d}  loss={loss_val:.4f}  lr={lr:.2e}")

            if global_step % args.val_every == 0:
                model.eval()
                val_losses = []
                t_levels = [0.5, 0.8]
                non_air_buckets = {t: [] for t in t_levels}
                common_buckets = {t: [] for t in t_levels}
                rare_buckets = {t: [] for t in t_levels}
                with torch.no_grad():
                    for val_batch in val_loader:
                        vb = val_batch["blocks"].to(device)
                        vc = val_batch["condition_mask"].to(device)
                        vv = val_batch["valid_mask"].to(device)
                        val_losses.append(
                            compute_loss(model, vb, vc, args.air_weight, vv).item()
                        )
                        for t in t_levels:
                            non_air, common, rare = compute_accuracy(model, vb, vc, t)
                            non_air_buckets[t].append(non_air)
                            common_buckets[t].append(common)
                            rare_buckets[t].append(rare)

                def avg(lst):
                    return sum(lst) / max(1, len(lst))

                avg_val = avg(val_losses)
                lines = [f"  val_loss={avg_val:.4f}"]
                for t in t_levels:
                    lines.append(
                        f"  t={t:.1f}  non_air={avg(non_air_buckets[t]):.3f}"
                        f"  common={avg(common_buckets[t]):.3f}"
                        f"  rare={avg(rare_buckets[t]):.3f}"
                    )
                tqdm.write("\n".join(lines))

                viz_path = os.path.join(run_dir, f"viz_step{global_step:07d}.png")
                save_sample_viz(
                    ema_model,
                    fixed_viz_blocks[:1],
                    fixed_viz_mask[:1],
                    viz_path,
                    global_step,
                )

                viz3d_path = os.path.join(run_dir, f"viz3d_step{global_step:07d}.html")
                save_3d_viz(
                    ema_model,
                    fixed_viz_blocks,
                    fixed_viz_mask,
                    viz3d_path,
                    global_step,
                    dataset.idx_to_block,
                )
                tqdm.write(f"  saved viz: {viz_path}  {viz3d_path}")

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
