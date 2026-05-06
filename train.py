import argparse
import contextlib
import copy
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MinecraftDataset
from diffusion import compute_loss, compute_accuracy
from model import UNetTransformer
from viz import save_sample_viz, save_3d_viz


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--min_fill", type=float, default=0.02)
    p.add_argument("--val_fraction", type=float, default=0.05)
    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--air_weight", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--save_every", type=int, default=5000)
    # model
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--transformer_layers", type=int, default=8)
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--time_dim", type=int, default=256)
    # infra
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    return p.parse_args()


def cosine_schedule(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_epoch(
    model, ema_model, loader, optim, sched, scaler, device, args, step, epoch, save_fn
):
    model.train()
    total_loss = 0.0
    n = 0
    autocast = torch.amp.autocast("cuda") if scaler else contextlib.nullcontext()

    bar = tqdm(loader, desc=f"  train", leave=False)
    for batch in bar:
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        with autocast:
            loss = compute_loss(model, blocks, cond, args.air_weight, valid)

        optim.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        sched.step()

        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.lerp_(p, 1.0 - args.ema_decay)

        lv = loss.item()
        total_loss += lv
        n += 1
        step += 1
        bar.set_postfix(loss=f"{lv:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        if step % args.save_every == 0:
            save_fn(step, epoch)

        if step >= args.max_steps:
            break

    return total_loss / max(1, n), step


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = []
    t_levels = (0.5, 0.8)
    buckets = {t: {"non_air": [], "common": [], "rare": []} for t in t_levels}

    for batch in tqdm(loader, desc=f"  val  ", leave=False):
        blocks = batch["blocks"].to(device, non_blocking=True)
        cond = batch["condition_mask"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)

        losses.append(compute_loss(model, blocks, cond, args.air_weight, valid).item())

        for t, b in buckets.items():
            na, co, ra = compute_accuracy(model, blocks, cond, t)
            b["non_air"].append(na)
            b["common"].append(co)
            b["rare"].append(ra)

    def avg(lst):
        return sum(lst) / max(1, len(lst))

    return {
        "loss": avg(losses),
        **{f"t{t}": {k: avg(v) for k, v in b.items()} for t, b in buckets.items()},
    }


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
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

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False).eval()

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: cosine_schedule(s, args.warmup_steps, args.max_steps)
    )
    scaler = (
        torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "vocab.pkl", "wb") as f:
        pickle.dump(
            {
                "block_to_idx": dataset.block_to_idx,
                "idx_to_block": dataset.idx_to_block,
                "vocab_size": dataset.vocab_size,
                "mask_idx": dataset.mask_idx,
            },
            f,
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def save_checkpoint(current_step, current_epoch, name=None):
        tag = name or f"step{current_step:07d}"
        path = run_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "step": current_step,
                "epoch": current_epoch,
                "vocab_size": dataset.vocab_size,
                "args": vars(args),
            },
            path,
        )
        tqdm.write(f"  saved → {path.name}")

    # ── Resume ────────────────────────────────────────────────────────────────
    step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"New weights (random init): {missing}")
        if unexpected:
            print(f"Dropped weights: {unexpected}")
        ema_model.load_state_dict(ckpt.get("ema_model", ckpt["model"]), strict=False)
        step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        try:
            optim.load_state_dict(ckpt["optim"])
            sched.load_state_dict(ckpt["sched"])
        except ValueError, RuntimeError:
            print("Optimizer state incompatible — fresh optimizer")
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=lambda s: cosine_schedule(
                    s, args.warmup_steps, args.max_steps
                ),
                last_epoch=step - 1,
            )
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from step {step}, epoch {start_epoch}")

    # ── Fixed viz batch (loaded once, stays constant) ─────────────────────────
    _viz_batch = next(
        iter(DataLoader(val_set, batch_size=10, shuffle=False, num_workers=0))
    )
    viz_blocks = _viz_batch["blocks"].to(device)
    viz_cond = _viz_batch["condition_mask"].to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch = start_epoch
    while step < args.max_steps:
        print(f"\nEpoch {epoch}  (step {step}/{args.max_steps})")

        train_loss, step = train_epoch(
            model,
            ema_model,
            train_loader,
            optim,
            sched,
            scaler,
            device,
            args,
            step,
            epoch,
            save_fn=save_checkpoint,
        )

        metrics = validate(ema_model, val_loader, device, args)
        model.train()

        print(
            f"  train_loss={train_loss:.4f}  val_loss={metrics['loss']:.4f}"
            f"  |  t=0.5  non_air={metrics['t0.5']['non_air']:.3f}"
            f"  common={metrics['t0.5']['common']:.3f}"
            f"  rare={metrics['t0.5']['rare']:.3f}"
            f"  |  t=0.8  non_air={metrics['t0.8']['non_air']:.3f}"
        )

        viz_path = str(run_dir / f"viz_epoch{epoch:03d}.png")
        viz3d_path = str(run_dir / f"viz3d_epoch{epoch:03d}.html")
        save_sample_viz(ema_model, viz_blocks[:1], viz_cond[:1], viz_path, step)
        save_3d_viz(
            ema_model, viz_blocks, viz_cond, viz3d_path, step, dataset.idx_to_block
        )
        tqdm.write(f"  viz → {Path(viz_path).name}  {Path(viz3d_path).name}")

        save_checkpoint(step, epoch, name=f"epoch{epoch:03d}")
        epoch += 1

    save_checkpoint(step, epoch, name="final")
    print("Training complete.")


if __name__ == "__main__":
    main()
