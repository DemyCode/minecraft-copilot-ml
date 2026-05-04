import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.pow(10000.0, -torch.arange(0, half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(emb)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_get_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_get_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale_shift = self.time_proj(t_emb)[:, :, None, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, spatial_size: int = 4, time_dim: int = 256):
        super().__init__()
        self.spatial_size = spatial_size
        num_tokens = spatial_size ** 3
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.time_proj = nn.Linear(time_dim, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = x + self.pos_emb + self.time_proj(t_emb).unsqueeze(1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        return x


class UNetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        transformer_layers: int = 8,
        transformer_heads: int = 8,
        chunk_size: int = 32,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_idx = vocab_size

        ch = [base_channels * m for m in channel_mult]
        bottleneck_ch = ch[-1]
        bottleneck_spatial = chunk_size // (2 ** (len(channel_mult) - 1))

        self.block_emb = nn.Embedding(vocab_size + 1, embed_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.conv_in = nn.Conv3d(embed_dim + 1, ch[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        in_ch = ch[0]
        skips = []
        for i, out_ch in enumerate(ch):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock3D(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.enc_blocks.append(level_blocks)
            skips.append(in_ch)
            if i < len(ch) - 1:
                self.enc_downs.append(Downsample3D(in_ch))
            else:
                self.enc_downs.append(nn.Identity())

        self.bottleneck = TransformerBottleneck(
            bottleneck_ch, transformer_layers, transformer_heads, bottleneck_spatial, time_dim
        )

        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        rev_ch = list(reversed(ch))
        rev_skips = list(reversed(skips))

        for i in range(len(ch) - 1):
            up_in = rev_ch[i]
            skip_ch = rev_skips[i + 1]
            out_ch = rev_ch[i + 1]
            self.dec_ups.append(Upsample3D(up_in))
            level_blocks = nn.ModuleList()
            in_ch = up_in + skip_ch
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock3D(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.dec_blocks.append(level_blocks)

        self.norm_out = nn.GroupNorm(_get_groups(ch[0]), ch[0])
        self.conv_out = nn.Conv3d(ch[0], vocab_size, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        condition_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)

        emb = self.block_emb(x_t).permute(0, 4, 1, 2, 3)
        cond = condition_mask.float().unsqueeze(1)
        h = self.conv_in(torch.cat([emb, cond], dim=1))

        skips_out = []
        for i, (res_blocks, down) in enumerate(zip(self.enc_blocks, self.enc_downs)):
            for blk in res_blocks:
                h = blk(h, t_emb)
            skips_out.append(h)
            if i < len(self.enc_downs) - 1:
                h = down(h)

        h = self.bottleneck(h, t_emb)

        skips_out = list(reversed(skips_out))
        for i, (up, res_blocks) in enumerate(zip(self.dec_ups, self.dec_blocks)):
            h = up(h)
            h = torch.cat([h, skips_out[i + 1]], dim=1)
            for blk in res_blocks:
                h = blk(h, t_emb)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
