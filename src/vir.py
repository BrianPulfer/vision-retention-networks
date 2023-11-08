import torch
import torch.nn as nn

DEFAULT_ALPHA = 1.00


class Retention(nn.Module):
    def __init__(self, embed_dim, max_len, alpha=DEFAULT_ALPHA, parallel=True):
        super(Retention, self).__init__()
        self.dim = embed_dim
        self.max_len = max_len
        self.alpha = alpha
        self.parallel = parallel

        # Useful buffers
        self.register_buffer("dim_sqrt", torch.tensor(embed_dim**0.5))
        self.register_buffer(
            "decay_mask",
            torch.tensor(
                [[alpha ** (i - j) for j in range(max_len)] for i in range(max_len)]
            ),
        )
        self.register_buffer("causal_mask", torch.ones(max_len, max_len).tril())
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

    def forward_parallel(self, x):
        # Getting queries, keys, values
        bs, sl, d = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Causal and decay masking
        M = (self.causal_mask[:sl, :sl] * self.decay_mask[:sl, :sl]).repeat(bs, 1, 1)

        # Retention
        out = (q @ k.transpose(-1, -2) / self.dim_sqrt * M) @ v
        return out

    def forward_recurrent(self, x):
        # TODO: ...
        pass

    def forward(self, x, parallel=None):
        if parallel is None:
            parallel = self.parallel

        if parallel:
            return self.forward_parallel(x)
        else:
            return self.forward_recurrent(x)


class MultiHeadRetention(nn.Module):
    def __init__(self, heads, embed_dim, max_len, alpha=DEFAULT_ALPHA, parallel=True):
        super(MultiHeadRetention, self).__init__()
        self.heads = heads
        self.parallel = parallel

        self.heads = nn.ModuleList(
            [Retention(embed_dim, max_len, alpha) for _ in range(heads)]
        )
        self.ln = nn.LayerNorm(embed_dim * heads)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(embed_dim * heads, embed_dim)

    def forward(self, x, parallel=None):
        if parallel is None:
            parallel = self.parallel

        out = torch.cat([head(x, parallel) for head in self.heads], dim=-1)
        return self.linear(self.gelu(self.ln(out)))


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class ViRBlock(nn.Module):
    def __init__(
        self, heads, embed_dim, max_len, alpha=DEFAULT_ALPHA, parallel=True, dropout=0.1
    ):
        super(ViRBlock, self).__init__()
        self.parallel = parallel

        self.ln1 = nn.LayerNorm(embed_dim)
        self.retention = MultiHeadRetention(heads, embed_dim, max_len, alpha, parallel)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, parallel=None):
        x = self.dropout1(self.retention(self.ln1(x), parallel=parallel)) + x
        x = self.dropout2(self.mlp(self.ln2(x))) + x
        return x


class ViR(nn.Module):
    def __init__(
        self,
        out_dim=10,
        patch_size=14,
        depth=12,
        heads=12,
        embed_dim=768,
        max_len=257,
        alpha=DEFAULT_ALPHA,
        parallel=True,
        dropout=0.1,
    ):
        super(ViR, self).__init__()

        # Local parameters
        self.out_dim = 10
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.alpha = alpha
        self.parallel = parallel

        # Embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ViR blocks
        self.blocks = nn.ModuleList(
            [
                ViRBlock(heads, embed_dim, max_len, alpha, parallel, dropout)
                for _ in range(depth)
            ]
        )

        # Head
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, out_dim)

    def forward(self, x, parallel=None):
        if parallel is None:
            parallel = self.parallel

        # Patch embedding, positional embedding, CLS token
        x = self.patch_embed(x).permute(0, 2, 3, 1).flatten(1, 2)
        bs, sl = x.shape[:2]
        x = x + self.pos_embed.repeat(bs, 1, 1)[:, :sl]
        x = torch.cat(
            (x, self.class_token.repeat(bs, 1, 1)), dim=1
        )  # Important: CLS token goes last

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head on the CLS token
        x = self.linear(self.ln(x[:, -1]))

        return x
