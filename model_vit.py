import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    # sinusoidal embedding, same idea as positional encoding in transformers
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    x = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # predicts 6 values: (shift, scale, gate) for attn and mlp each
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        # attention sub-block
        h = self.norm1(x) * (1 + sc1[:, None]) + s1[:, None]
        attn_out, _ = self.attn(h, h, h)
        x = x + g1[:, None] * attn_out
        # mlp sub-block
        h = self.norm2(x) * (1 + sc2[:, None]) + s2[:, None]
        x = x + g2[:, None] * self.mlp(h)
        return x


class DiTFlow(nn.Module):
    # DiT-S/4: 256 patches, 512 dim, 8 layers
    def __init__(self, img_size=64, patch_size=4, in_ch=3, dim=512, depth=8, heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.img_size = img_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, dim))
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(dim, patch_size * patch_size * in_ch)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x):
        p, c = self.patch_size, self.in_ch
        h = w = self.img_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t):
        c = timestep_embedding(t * 1000.0, self.t_embed[0].in_features)
        c = self.t_embed(c)

        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x, c)

        # final layer with adaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale[:, None]) + shift[:, None]
        x = self.final_proj(x)
        return self.unpatchify(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
