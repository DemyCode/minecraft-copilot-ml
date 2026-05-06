import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from diffusion import sample


def save_sample_viz(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    path: str,
    step: int,
    t_levels: tuple = (0.2, 0.4, 0.6, 0.8),
    num_steps: int = 20,
):
    N = min(blocks.shape[0], 2)
    cs = blocks.shape[-1]
    mid = cs // 2
    vocab_size = model.vocab_size

    was_training = model.training
    model.eval()

    n_cols = 1 + len(t_levels)
    fig, axes = plt.subplots(N, n_cols, figsize=(n_cols * 3, N * 3 + 0.5))
    if N == 1:
        axes = axes[None]
    fig.suptitle(f"step {step}", fontsize=11)

    axes[0, 0].set_title("condition", fontsize=9)
    for j, t in enumerate(t_levels):
        axes[0, j + 1].set_title(f"recon t={t}", fontsize=9)

    imshow_kw = dict(cmap="tab20b", vmin=0, vmax=vocab_size, interpolation="nearest")

    for i in range(N):
        gt = blocks[i]
        mask = condition_mask[i]

        cond_vis = (gt * mask).cpu().numpy()
        axes[i, 0].imshow(cond_vis[mid], **imshow_kw)
        axes[i, 0].axis("off")

        for j, t in enumerate(t_levels):
            with torch.no_grad():
                result = sample(model, gt[None], mask[None], num_steps=num_steps, t_start=t)
            axes[i, j + 1].imshow(result[0, mid].cpu().numpy(), **imshow_kw)
            axes[i, j + 1].axis("off")

    if was_training:
        model.train()

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_3d_viz(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    path: str,
    step: int,
    idx_to_block: dict,
    t_levels: tuple = (0.2, 0.4, 0.6, 0.8),
    num_steps: int = 20,
):
    N = min(blocks.shape[0], 2)
    cs = blocks.shape[-1]

    was_training = model.training
    model.eval()

    all_traces = []
    buttons = []
    combo_idx = 0

    for i in range(N):
        gt = blocks[i]
        mask = condition_mask[i]
        mask_np = mask.cpu().numpy()

        for t in t_levels:
            with torch.no_grad():
                result = sample(model, gt[None], mask[None], num_steps=num_steps, t_start=t)
            arr = result[0].cpu().numpy()
            visible = combo_idx == 0

            cy, cz, cx = np.where(mask_np & (arr != 0))
            if len(cx):
                all_traces.append(go.Scatter3d(
                    x=cx, y=cy, z=cz,
                    mode="markers",
                    marker=dict(size=4, color="lightsteelblue", opacity=0.25),
                    name=f"s{i+1} t={t} hint",
                    visible=visible,
                    hovertext=[idx_to_block.get(int(arr[y, z, x]), "?") for y, z, x in zip(cy, cz, cx)],
                    hoverinfo="text+x+y+z",
                ))
            else:
                all_traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible, name=f"s{i+1} t={t} hint"))

            py, pz, px = np.where(~mask_np & (arr != 0))
            if len(px):
                hover = [idx_to_block.get(int(arr[y, z, x]), "?") for y, z, x in zip(py, pz, px)]
                all_traces.append(go.Scatter3d(
                    x=px, y=py, z=pz,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=arr[py, pz, px].tolist(),
                        colorscale="Rainbow",
                        cmin=0,
                        cmax=model.vocab_size,
                        opacity=0.35,
                        colorbar=dict(title="block idx", thickness=12),
                    ),
                    name=f"s{i+1} t={t} recon",
                    visible=visible,
                    hovertext=hover,
                    hoverinfo="text+x+y+z",
                ))
            else:
                all_traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible, name=f"s{i+1} t={t} recon"))

            combo_idx += 1

    n_combos = N * len(t_levels)
    traces_per_combo = 2
    for k in range(n_combos):
        vis = [False] * (n_combos * traces_per_combo)
        vis[k * traces_per_combo] = True
        vis[k * traces_per_combo + 1] = True
        i, j = divmod(k, len(t_levels))
        buttons.append(dict(
            label=f"S{i+1} t={t_levels[j]}",
            method="update",
            args=[{"visible": vis}],
        ))

    if was_training:
        model.train()

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"step {step}",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )],
        scene=dict(
            xaxis=dict(range=[0, cs], title="x"),
            yaxis=dict(range=[0, cs], title="y"),
            zaxis=dict(range=[0, cs], title="z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn")
