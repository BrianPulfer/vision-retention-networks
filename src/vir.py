import torch
import torch.nn as nn

DEFAULT_ALPHA = 1.00


class ViRModes:
    PARALLEL = "parallel"
    RECURRENT = "recurrent"
    CHUNKWISE_PARALLEL = "chunkwise_parallel"


class Retention(nn.Module):
    def __init__(self, embed_dim, max_len, alpha=DEFAULT_ALPHA, mode=ViRModes.PARALLEL):
        super(Retention, self).__init__()
        self.dim = embed_dim
        self.max_len = max_len
        self.alpha = alpha
        self.mode = mode

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

    def forward_recurrent(self, x, state):
        assert x.ndim == 2, "Recurrent mode only works with 2D inputs"

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        new_state = self.alpha * state + k.unsqueeze(-1) @ v.unsqueeze(1)
        out = q.unsqueeze(1) @ new_state / self.dim_sqrt

        return out.squeeze(), new_state

    def forward(self, x, state=None, mode=ViRModes.PARALLEL):
        if mode is None:
            mode = self.mode

        if mode == ViRModes.PARALLEL:
            return self.forward_parallel(x)
        elif mode == ViRModes.RECURRENT:
            return self.forward_recurrent(x, state)


class MultiHeadRetention(nn.Module):
    def __init__(
        self, heads, embed_dim, max_len, alpha=DEFAULT_ALPHA, mode=ViRModes.PARALLEL
    ):
        super(MultiHeadRetention, self).__init__()
        self.n_heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.mode = mode

        assert (
            embed_dim % heads == 0
        ), "Embedding dimension must be divisible by the number of heads"

        self.heads = nn.ModuleList(
            [Retention(embed_dim // heads, max_len, alpha) for _ in range(heads)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode

        if mode == ViRModes.PARALLEL:
            return self.forward_parallel(x)
        elif mode == ViRModes.RECURRENT:
            return self.forward_recurrent(x)

    def forward_parallel(self, x):
        out = torch.cat(
            [
                head(
                    x[:, :, i * self.head_dim : (i + 1) * self.head_dim],
                    mode=ViRModes.PARALLEL,
                )
                for i, head in enumerate(self.heads)
            ],
            dim=-1,
        )
        return self.linear(self.gelu(self.ln(out)))

    def forward_recurrent(self, x):
        """Given a B, L, D input, returns a B, D output (last token output)"""
        batch_size, length, dim = x.shape

        all_outputs = []
        state = [
            torch.zeros(batch_size, dim // self.n_heads, dim // self.n_heads).cuda()
            for _ in range(self.n_heads)
        ]
        for i in range(length):
            outs_and_states = [
                head(
                    x[:, i, j * self.head_dim : (j + 1) * self.head_dim],
                    state[j],
                    mode=ViRModes.RECURRENT,
                )
                for j, head in enumerate(self.heads)
            ]

            all_outputs.append(torch.cat([out for out, _ in outs_and_states], dim=1))
            state = [new_state for _, new_state in outs_and_states]

        x = torch.stack(all_outputs, dim=1)
        return self.linear(self.gelu(self.ln(x)))


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
        self,
        heads,
        embed_dim,
        max_len,
        alpha=DEFAULT_ALPHA,
        mode=ViRModes.PARALLEL,
        dropout=0.1,
    ):
        super(ViRBlock, self).__init__()
        self.mode = mode

        self.ln1 = nn.LayerNorm(embed_dim)
        self.retention = MultiHeadRetention(heads, embed_dim, max_len, alpha, mode)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode

        x = self.dropout1(self.retention(self.ln1(x), mode=mode)) + x
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
        mode=ViRModes.PARALLEL,
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
        self.mode = mode

        # Embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ViR blocks
        self.blocks = nn.ModuleList(
            [
                ViRBlock(heads, embed_dim, max_len, alpha, mode, dropout)
                for _ in range(depth)
            ]
        )

        # Head
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, out_dim)

    def set_compute_mode(self, mode):
        self.mode = mode

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode

        # Patch embedding, positional embedding, CLS token
        x = self.patch_embed(x).permute(0, 2, 3, 1).flatten(1, 2)
        bs, sl = x.shape[:2]
        x = x + self.pos_embed.repeat(bs, 1, 1)[:, :sl]
        x = torch.cat(
            (x, self.class_token.repeat(bs, 1, 1)), dim=1
        )  # Important: CLS token goes last

        # Blocks
        for block in self.blocks:
            x = block(x, mode=mode)

        # Head on the CLS token
        x = self.linear(self.ln(x[:, -1]))

        return x


if __name__ == "__main__":
    # Testing that parallel and recurrent modes give the same output

    model = ViR(depth=12, heads=3, embed_dim=192).eval().cuda()
    x = torch.randn(16, 3, 224, 224).cuda()

    with torch.no_grad():
        model.set_compute_mode(ViRModes.PARALLEL)
        y1 = model(x)

        model.set_compute_mode(ViRModes.RECURRENT)
        y2 = model(x)

        assert torch.allclose(
            y1, y2, atol=1e-6
        ), "Parallel and recurrent modes should give the same output"
    print("Test passed! (parallel and recurrent modes give the same output)")
