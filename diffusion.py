import torch
import torch.nn.functional as F


def compute_loss(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    condition_mask: torch.Tensor,
    air_weight: float = 0.1,
) -> torch.Tensor:
    B = blocks.shape[0]
    device = blocks.device

    t = torch.rand(B, device=device)

    unknown = ~condition_mask
    absorb_prob = t[:, None, None, None].expand_as(blocks.float())
    noise_mask = (torch.rand_like(blocks.float()) < absorb_prob) & unknown

    if not noise_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    x_t = blocks.clone()
    x_t[noise_mask] = model.mask_idx

    logits = model(x_t, condition_mask, t)

    vocab_size = logits.shape[1]
    weight = torch.ones(vocab_size, device=device)
    weight[0] = air_weight

    loss = F.cross_entropy(
        logits.permute(0, 2, 3, 4, 1)[noise_mask],
        blocks[noise_mask],
        weight=weight,
    )

    return loss


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    device = condition.device
    B = condition.shape[0]
    mask_idx = model.mask_idx
    vocab_size = model.vocab_size

    x = condition.clone()
    x[~condition_mask] = mask_idx

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for step in range(num_steps):
        t_now = t_steps[step].item()
        t_next = t_steps[step + 1].item()

        t_tensor = torch.full((B,), t_now, device=device)
        logits = model(x, condition_mask, t_tensor)

        flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        if temperature != 1.0:
            flat_logits = flat_logits / temperature
        probs = F.softmax(flat_logits, dim=-1)
        x0_flat = torch.multinomial(probs, 1).squeeze(1)
        x0 = x0_flat.reshape(B, *condition.shape[1:])

        still_masked = (x == mask_idx) & ~condition_mask
        if not still_masked.any():
            break

        if t_now > 1e-6:
            unmask_prob = (t_now - t_next) / t_now
            should_unmask = (torch.rand_like(x.float()) < unmask_prob) & still_masked
        else:
            should_unmask = still_masked

        x[should_unmask] = x0[should_unmask]

    remaining = (x == mask_idx) & ~condition_mask
    if remaining.any():
        t_zero = torch.zeros(B, device=device)
        logits = model(x, condition_mask, t_zero)
        x0_final = logits.argmax(dim=1)
        x[remaining] = x0_final[remaining]

    return x


@torch.no_grad()
def sample_progressive(
    model: torch.nn.Module,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
    yield_every: int = 5,
):
    device = condition.device
    B = condition.shape[0]
    mask_idx = model.mask_idx
    vocab_size = model.vocab_size

    x = condition.clone()
    x[~condition_mask] = mask_idx

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for step in range(num_steps):
        t_now = t_steps[step].item()
        t_next = t_steps[step + 1].item()

        t_tensor = torch.full((B,), t_now, device=device)
        logits = model(x, condition_mask, t_tensor)

        flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, vocab_size)
        if temperature != 1.0:
            flat_logits = flat_logits / temperature
        probs = F.softmax(flat_logits, dim=-1)
        x0_flat = torch.multinomial(probs, 1).squeeze(1)
        x0 = x0_flat.reshape(B, *condition.shape[1:])

        still_masked = (x == mask_idx) & ~condition_mask
        if not still_masked.any():
            break

        if t_now > 1e-6:
            unmask_prob = (t_now - t_next) / t_now
            should_unmask = (torch.rand_like(x.float()) < unmask_prob) & still_masked
        else:
            should_unmask = still_masked

        x[should_unmask] = x0[should_unmask]

        if step % yield_every == 0 or step == num_steps - 1:
            yield x.clone()

    remaining = (x == mask_idx) & ~condition_mask
    if remaining.any():
        t_zero = torch.zeros(B, device=device)
        logits = model(x, condition_mask, t_zero)
        x0_final = logits.argmax(dim=1)
        x[remaining] = x0_final[remaining]
        yield x.clone()
