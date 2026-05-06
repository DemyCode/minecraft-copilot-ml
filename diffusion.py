import torch
import torch.nn.functional as F


def compute_loss(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    air_weight: float = 0.1,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B = blocks.shape[0]
    device = blocks.device

    t = torch.rand(B, device=device)
    blocks_f = blocks.float()

    unknown = ~condition_mask
    absorb_prob = t[:, None, None, None].expand_as(blocks_f)
    noise_mask = (torch.rand_like(blocks_f) < absorb_prob) & unknown

    x_t = blocks.clone()
    x_t[noise_mask] = model.mask_idx

    logits = model(x_t, condition_mask, t)

    # Restrict loss to valid (non-padded) positions
    loss_mask = noise_mask & valid_mask if valid_mask is not None else noise_mask

    if not loss_mask.any():
        return logits.sum() * 0.0

    vocab_size = logits.shape[1]
    weight = torch.ones(vocab_size, device=device)
    weight[0] = air_weight

    return F.cross_entropy(
        logits.permute(0, 2, 3, 4, 1)[loss_mask],
        blocks[loss_mask],
        weight=weight,
    )


@torch.no_grad()
def compute_accuracy(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    t: float,
    common_cutoff: int = 10,
) -> tuple[float, float, float]:
    B = blocks.shape[0]
    device = blocks.device

    t_tensor = torch.full((B,), t, device=device)
    blocks_f = blocks.float()
    unknown = ~condition_mask
    absorb_prob = t_tensor[:, None, None, None].expand_as(blocks_f)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        noise_mask = (torch.rand_like(blocks_f) < absorb_prob) & unknown

    if not noise_mask.any():
        return 0.0, 0.0, 0.0

    x_t = blocks.clone()
    x_t[noise_mask] = model.mask_idx

    logits = model(x_t, condition_mask, t_tensor)
    preds = logits.argmax(dim=1)

    non_air_mask = noise_mask & (blocks != 0)
    non_air_acc = (preds[non_air_mask] == blocks[non_air_mask]).float().mean().item() if non_air_mask.any() else 0.0

    common_mask = noise_mask & (blocks > 0) & (blocks <= common_cutoff)
    common_acc = (preds[common_mask] == blocks[common_mask]).float().mean().item() if common_mask.any() else 0.0

    rare_mask = noise_mask & (blocks > common_cutoff)
    rare_acc = (preds[rare_mask] == blocks[rare_mask]).float().mean().item() if rare_mask.any() else 0.0

    return non_air_acc, common_acc, rare_acc


def _sample_steps(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int,
    temperature: float,
    t_start: float = 1.0,
):
    device = condition.device
    B = condition.shape[0]
    mask_idx = model.mask_idx
    vocab_size = model.vocab_size

    x = condition.clone()
    unknown = ~condition_mask
    if t_start >= 1.0:
        x[unknown] = mask_idx
    else:
        noise = torch.rand_like(x.float())
        x[unknown & (noise < t_start)] = mask_idx

    t_steps = torch.linspace(t_start, 0.0, num_steps + 1, device=device)

    for step in range(num_steps):
        still_masked = (x == mask_idx) & ~condition_mask
        if not still_masked.any():
            break

        t_now = t_steps[step].item()
        t_next = t_steps[step + 1].item()

        t_tensor = torch.full((B,), t_now, device=device)
        logits = model(x, condition_mask, t_tensor)

        flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        if temperature != 1.0:
            flat_logits = flat_logits / temperature
        probs = F.softmax(flat_logits, dim=-1)
        x0 = torch.multinomial(probs, 1).squeeze(1).reshape(B, *condition.shape[1:])

        if t_now > 1e-6:
            unmask_prob = (t_now - t_next) / t_now
            should_unmask = (torch.rand_like(x.float()) < unmask_prob) & still_masked
        else:
            should_unmask = still_masked

        x[should_unmask] = x0[should_unmask]
        yield step, x

    remaining = (x == mask_idx) & ~condition_mask
    if remaining.any():
        t_zero = torch.zeros(B, device=device)
        logits = model(x, condition_mask, t_zero)
        x[remaining] = logits.argmax(dim=1)[remaining]

    yield -1, x


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
    t_start: float = 1.0,
) -> torch.Tensor:
    for _, x in _sample_steps(model, condition, condition_mask, num_steps, temperature, t_start):
        pass
    return x


@torch.no_grad()
def sample_progressive(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
    yield_every: int = 5,
    t_start: float = 1.0,
):
    for step, x in _sample_steps(model, condition, condition_mask, num_steps, temperature, t_start):
        if step == -1 or step % yield_every == 0 or step == num_steps - 1:
            yield x.clone()
