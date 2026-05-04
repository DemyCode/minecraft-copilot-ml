import argparse
import os
import pickle

import numpy as np
import torch

from diffusion import sample, sample_progressive
from model import UNetTransformer


def load_model_and_vocab(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    run_dir = os.path.dirname(checkpoint_path)
    vocab_path = os.path.join(run_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    args = ckpt.get("args", {})

    model = UNetTransformer(
        vocab_size=vocab["vocab_size"],
        embed_dim=args.get("embed_dim", 128),
        base_channels=args.get("base_channels", 64),
        num_res_blocks=args.get("num_res_blocks", 2),
        time_dim=args.get("time_dim", 256),
        transformer_layers=args.get("transformer_layers", 8),
        transformer_heads=args.get("transformer_heads", 8),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, vocab


def complete_structure(
    checkpoint_path: str,
    placed_blocks: np.ndarray,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
    progressive: bool = False,
    yield_every: int = 5,
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, vocab = load_model_and_vocab(checkpoint_path, dev)

    block_to_idx = vocab["block_to_idx"]
    idx_to_block = vocab["idx_to_block"]

    cs = placed_blocks.shape[0]
    assert placed_blocks.shape == (cs, cs, cs), "placed_blocks must be a cube"

    known_mask = placed_blocks != None
    condition_int = np.zeros((cs, cs, cs), dtype=np.int64)
    for idx in np.ndindex(cs, cs, cs):
        if known_mask[idx]:
            name = str(placed_blocks[idx])
            condition_int[idx] = block_to_idx.get(name, block_to_idx.get("minecraft:air", 0))

    condition = torch.from_numpy(condition_int).unsqueeze(0).to(dev)
    mask = torch.from_numpy(known_mask).unsqueeze(0).to(dev)

    def decode(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.squeeze(0).cpu().numpy()
        result = np.empty((cs, cs, cs), dtype=object)
        for i in np.ndindex(cs, cs, cs):
            result[i] = idx_to_block.get(int(arr[i]), "minecraft:air")
        return result

    if progressive:
        def gen():
            for step_tensor in sample_progressive(
                model, condition, mask, num_steps, temperature, yield_every
            ):
                yield decode(step_tensor)
        return gen()

    result = sample(model, condition, mask, num_steps, temperature)
    return decode(result)


def save_schematic(block_names: np.ndarray, output_path: str):
    import nbtlib
    from nbtlib import File, Compound, Int, IntArray

    h, l, w = block_names.shape
    palette: dict = {}
    block_data = []

    for y in range(h):
        for z in range(l):
            for x in range(w):
                name = str(block_names[y, z, x])
                if name not in palette:
                    palette[name] = len(palette)
                block_data.append(palette[name])

    schematic = Compound(
        {
            "Version": Int(2),
            "DataVersion": Int(2975),
            "Width": Int(w),
            "Height": Int(h),
            "Length": Int(l),
            "Palette": Compound({k: Int(v) for k, v in palette.items()}),
            "BlockData": IntArray(block_data),
        }
    )

    File({"Schematic": schematic}).save(output_path)
    print(f"Saved schematic: {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="output.schem")
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Path to a .npy file of shape [cs, cs, cs] with block name strings (None = unknown)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cs = args.chunk_size

    if args.condition is not None:
        placed = np.load(args.condition, allow_pickle=True)
    else:
        placed = np.full((cs, cs, cs), None, dtype=object)
        placed[: cs // 2] = "minecraft:stone"

    result = complete_structure(
        checkpoint_path=args.checkpoint,
        placed_blocks=placed,
        num_steps=args.num_steps,
        temperature=args.temperature,
        device=args.device,
    )

    save_schematic(result, args.output)


if __name__ == "__main__":
    main()
