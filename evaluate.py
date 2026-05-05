import argparse
import os
import pickle

import numpy as np
import plotly.graph_objects as go
import torch

from diffusion import sample
from model import UNetTransformer
from schematic_loader import load_schematic


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(run_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    a = ckpt.get("args", {})
    model = UNetTransformer(
        vocab_size=vocab["vocab_size"],
        embed_dim=a.get("embed_dim", 128),
        base_channels=a.get("base_channels", 64),
        num_res_blocks=a.get("num_res_blocks", 2),
        time_dim=a.get("time_dim", 256),
        transformer_layers=a.get("transformer_layers", 8),
        transformer_heads=a.get("transformer_heads", 8),
        chunk_size=a.get("chunk_size", 32),
    ).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.eval()
    cs = a.get("chunk_size", 32)
    return model, vocab, cs


def to_indices(blocks_str: np.ndarray, block_to_idx: dict) -> np.ndarray:
    unique, inverse = np.unique(blocks_str, return_inverse=True)
    idx = np.array([block_to_idx.get(str(n), 0) for n in unique], dtype=np.int64)
    return idx[inverse].reshape(blocks_str.shape)


def extract_chunk(blocks_str: np.ndarray, cs: int) -> np.ndarray:
    chunk = np.full((cs, cs, cs), "minecraft:air", dtype=object)
    h, l, w = blocks_str.shape

    def center_slice(dim_size):
        if dim_size >= cs:
            s = (dim_size - cs) // 2
            return slice(s, s + cs), slice(0, cs)
        return slice(0, dim_size), slice(0, dim_size)

    y_src, y_dst = center_slice(h)
    z_src, z_dst = center_slice(l)
    x_src, x_dst = center_slice(w)
    chunk[y_dst, z_dst, x_dst] = blocks_str[y_src, z_src, x_src]
    return chunk


def make_mask(cs: int, strategy: str, cut_frac: float) -> np.ndarray:
    cut = int(cs * cut_frac)
    mask = np.zeros((cs, cs, cs), dtype=bool)
    if strategy == "bottom":
        mask[:cut] = True
    elif strategy == "top":
        mask[cut:] = True
    elif strategy == "random":
        mask = np.random.rand(cs, cs, cs) < cut_frac
    elif strategy == "shell":
        t = max(1, cs // 8)
        mask[:t] = mask[-t:] = True
        mask[:, :t] = mask[:, -t:] = True
        mask[:, :, :t] = mask[:, :, -t:] = True
    return mask


def metrics(original: np.ndarray, reconstructed: np.ndarray, condition_mask: np.ndarray, common_cutoff: int = 10) -> dict:
    unknown = ~condition_mask
    if not unknown.any():
        return {}
    o, r = original[unknown], reconstructed[unknown]
    exact = (o == r).mean()

    non_air = unknown & (original != 0)
    non_air_acc = (original[non_air] == reconstructed[non_air]).mean() if non_air.any() else float("nan")

    common = unknown & (original > 0) & (original <= common_cutoff)
    common_acc = (original[common] == reconstructed[common]).mean() if common.any() else float("nan")

    rare = unknown & (original > common_cutoff)
    rare_acc = (original[rare] == reconstructed[rare]).mean() if rare.any() else float("nan")

    hallucinated = unknown & (original == 0) & (reconstructed != 0)
    missed = unknown & (original != 0) & (reconstructed == 0)

    return {
        "exact": float(exact),
        "non_air": float(non_air_acc),
        "common": float(common_acc),
        "rare": float(rare_acc),
        "hallucinated_frac": float(hallucinated.sum() / max(1, (unknown & (original == 0)).sum())),
        "missed_frac": float(missed.sum() / max(1, non_air.sum())),
        "n_unknown": int(unknown.sum()),
        "n_non_air_unknown": int(non_air.sum()),
    }


def save_viz(original: np.ndarray, condition_mask: np.ndarray, reconstructed: np.ndarray,
             path: str, idx_to_block: dict):
    cs = original.shape[0]
    cmax = max(int(original.max()), int(reconstructed.max()), 1)
    all_traces = []

    def scatter(arr, boolean_mask, name, colorscale, opacity, visible):
        ys, zs, xs = np.where(boolean_mask & (arr != 0))
        if not len(xs):
            return go.Scatter3d(x=[], y=[], z=[], visible=visible, name=name)
        hover = [idx_to_block.get(int(arr[y, z, x]), "?") for y, z, x in zip(ys, zs, xs)]
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=4, color=arr[ys, zs, xs].tolist(),
                        colorscale=colorscale, cmin=0, cmax=cmax, opacity=opacity),
            name=name,
            visible=visible,
            hovertext=hover,
            hoverinfo="text+x+y+z",
        )

    unknown = ~condition_mask

    all_traces.append(scatter(original, np.ones_like(original, dtype=bool), "original", "Rainbow", 0.5, True))
    all_traces.append(scatter(original, condition_mask, "input (given)", "Rainbow", 0.6, False))
    all_traces.append(scatter(reconstructed, np.ones_like(reconstructed, dtype=bool), "reconstructed", "Rainbow", 0.5, False))

    correct_non_air = unknown & (original != 0) & (original == reconstructed)
    wrong_non_air = unknown & (original != 0) & (original != reconstructed)
    hallucinated = unknown & (original == 0) & (reconstructed != 0)
    missed = unknown & (original != 0) & (reconstructed == 0)

    diff_traces = []
    for mask, color, label in [
        (correct_non_air, "green", "correct"),
        (wrong_non_air, "red", "wrong type"),
        (hallucinated, "orange", "hallucinated"),
        (missed, "blue", "missed"),
    ]:
        ys, zs, xs = np.where(mask)
        if len(xs):
            diff_traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.7),
                name=f"diff: {label} ({len(xs)})",
                visible=False,
            ))

    all_traces.extend(diff_traces)

    n_base = 3
    n_diff = len(diff_traces)

    def vis_flags(active_indices):
        flags = [False] * len(all_traces)
        for i in active_indices:
            if i < len(all_traces):
                flags[i] = True
        return flags

    buttons = [
        dict(label="original", method="update", args=[{"visible": vis_flags([0])}]),
        dict(label="input (given)", method="update", args=[{"visible": vis_flags([1])}]),
        dict(label="reconstructed", method="update", args=[{"visible": vis_flags([2])}]),
        dict(label="diff", method="update", args=[{"visible": vis_flags(list(range(n_base, n_base + n_diff)))}]),
    ]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title="Reconstruction evaluation",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0, xanchor="left", y=1.15, yanchor="top")],
        scene=dict(
            xaxis=dict(range=[0, cs], title="x"),
            yaxis=dict(range=[0, cs], title="y"),
            zaxis=dict(range=[0, cs], title="z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--schematic", required=True)
    p.add_argument("--mask", default="bottom", choices=["bottom", "top", "random", "shell"])
    p.add_argument("--cut_frac", type=float, default=0.5)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--output", default="eval.html")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, vocab, cs = load_model(args.checkpoint, device)
    block_to_idx = vocab["block_to_idx"]
    idx_to_block = vocab["idx_to_block"]

    print(f"Loading: {args.schematic}")
    blocks_str = load_schematic(args.schematic)
    h, l, w = blocks_str.shape
    print(f"Schematic size: {h}×{l}×{w}  →  center-cropped to {cs}×{cs}×{cs}")

    chunk_str = extract_chunk(blocks_str, cs)
    original = to_indices(chunk_str, block_to_idx)
    fill = (original != 0).mean()
    print(f"Fill ratio: {fill:.1%}  |  unique blocks: {len(np.unique(original))}")

    condition_mask = make_mask(cs, args.mask, args.cut_frac)
    known_pct = condition_mask.mean()
    print(f"Mask: {args.mask}  given={known_pct:.0%}  to-reconstruct={1-known_pct:.0%}")

    orig_t = torch.from_numpy(original).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(condition_mask).unsqueeze(0).to(device)

    all_metrics = []
    last_result = None
    for run in range(args.n_runs):
        print(f"  Run {run+1}/{args.n_runs}...", end=" ", flush=True)
        with torch.no_grad():
            result_t = sample(model, orig_t, mask_t, num_steps=args.num_steps, temperature=args.temperature)
        result = result_t.squeeze(0).cpu().numpy()
        m = metrics(original, result, condition_mask)
        all_metrics.append(m)
        last_result = result
        print(f"rare={m['rare']:.3f}  non_air={m['non_air']:.3f}")

    print("\n--- Metrics at unknown positions ---")
    for key in ["exact", "non_air", "common", "rare", "hallucinated_frac", "missed_frac"]:
        vals = [m[key] for m in all_metrics]
        mean = np.mean(vals)
        if args.n_runs > 1:
            std = np.std(vals)
            print(f"  {key:<22} {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {key:<22} {mean:.3f}")
    print(f"  {'n_unknown':<22} {all_metrics[0]['n_unknown']}")
    print(f"  {'n_non_air_unknown':<22} {all_metrics[0]['n_non_air_unknown']}")

    print(f"\nSaving viz: {args.output}")
    save_viz(original, condition_mask, last_result, args.output, idx_to_block)
    print("Done.")


if __name__ == "__main__":
    main()
