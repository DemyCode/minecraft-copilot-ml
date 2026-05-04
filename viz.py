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
    num_steps: int = 20,
):
    device = blocks.device
    was_training = model.training
    model.eval()

    with torch.no_grad():
        result = sample(model, blocks, condition_mask, num_steps=num_steps)

    if was_training:
        model.train()

    cs = blocks.shape[-1]
    mid = cs // 2
    vocab_size = model.vocab_size

    gt = blocks[0].cpu().numpy()
    res = result[0].cpu().numpy()
    mask = condition_mask[0].cpu().numpy()

    cond_vis = gt * mask

    slices = [
        (cond_vis[mid], res[mid], "horizontal"),
        (cond_vis[:, mid, :], res[:, mid, :], "depth"),
        (cond_vis[:, :, mid], res[:, :, mid], "side"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"step {step}", fontsize=12)

    for col, (cond_s, res_s, title) in enumerate(slices):
        axes[0, col].imshow(cond_s, cmap="tab20b", vmin=0, vmax=vocab_size, interpolation="nearest")
        axes[0, col].set_title(f"input ({title})")
        axes[0, col].axis("off")

        axes[1, col].imshow(res_s, cmap="tab20b", vmin=0, vmax=vocab_size, interpolation="nearest")
        axes[1, col].set_title(f"predicted ({title})")
        axes[1, col].axis("off")

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
    num_steps: int = 20,
):
    # blocks: [N, cs, cs, cs], condition_mask: [N, cs, cs, cs]
    N = blocks.shape[0]
    cs = blocks.shape[-1]
    device = blocks.device

    was_training = model.training
    model.eval()
    with torch.no_grad():
        result = sample(model, blocks, condition_mask, num_steps=num_steps)
    if was_training:
        model.train()

    all_traces = []

    for i in range(N):
        arr = result[i].cpu().numpy()
        mask = condition_mask[i].cpu().numpy()
        visible = i == 0

        cy, cz, cx = np.where(mask & (arr != 0))
        if len(cx):
            all_traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="markers",
                marker=dict(size=4, color="lightsteelblue", opacity=0.25),
                name=f"#{i+1} condition",
                visible=visible,
                hovertext=[idx_to_block.get(int(arr[y, z, x]), "?") for y, z, x in zip(cy, cz, cx)],
                hoverinfo="text+x+y+z",
            ))
        else:
            all_traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible, name=f"#{i+1} condition (empty)"))

        py, pz, px = np.where(~mask & (arr != 0))
        if len(px):
            colors = arr[py, pz, px].tolist()
            hover = [idx_to_block.get(int(arr[y, z, x]), "?") for y, z, x in zip(py, pz, px)]
            all_traces.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale="Rainbow",
                    cmin=0,
                    cmax=model.vocab_size,
                    opacity=0.35,
                    colorbar=dict(title="block idx", thickness=12),
                ),
                name=f"#{i+1} predicted",
                visible=visible,
                hovertext=hover,
                hoverinfo="text+x+y+z",
            ))
        else:
            all_traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible, name=f"#{i+1} predicted (empty)"))

    traces_per_sample = 2
    buttons = []
    for i in range(N):
        vis = [False] * (N * traces_per_sample)
        vis[i * traces_per_sample] = True
        vis[i * traces_per_sample + 1] = True
        buttons.append(dict(label=f"Sample {i+1}", method="update", args=[{"visible": vis}]))

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
