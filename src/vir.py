import torch
import torch.nn as nn

DEFAULT_ALPHA = 1.00


class ViRModes:
    PARALLEL = "parallel"
    RECURRENT = "recurrent"
    CHUNKWISE = "chunkwise"


class Retention(nn.Module):
    def __init__(
        self,
        embed_dim,
        max_len,
        alpha=DEFAULT_ALPHA,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
    ):
        super(Retention, self).__init__()
        self.dim = embed_dim
        self.max_len = max_len
        self.chunk_size = chunk_size
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
        batch_size, length, dim = x.shape

        all_outputs = []
        state = torch.zeros(batch_size, dim, dim).to(x.device)
        for i in range(length):
            xi = x[:, i]
            q, k, v = self.qkv(xi).chunk(3, dim=-1)

            state = self.alpha * state + k.unsqueeze(-1) @ v.unsqueeze(1)
            out = q.unsqueeze(1) @ state / self.dim_sqrt
            all_outputs.append(out.squeeze())

        x = torch.stack(all_outputs, dim=1)
        return x

    def forward_chunkwise(self, x, chunk_size=None):
        # Getting queries, keys, values
        if chunk_size is None:
            chunk_size = self.chunk_size

        bs, sl, d = x.shape

        outputs = []
        state = torch.zeros(bs, d, d).to(x.device)
        for xi in [x[:, i : i + chunk_size] for i in range(0, sl, chunk_size)]:
            qkv = self.qkv(xi)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            # Masking matrices
            cl = xi.shape[1]
            retention_mask = (
                torch.tensor(
                    [
                        [self.alpha ** (chunk_size - i - 1) for _ in range(d)]
                        for i in range(cl)
                    ]
                )
                .repeat(bs, 1, 1)
                .to(x.device)
            )
            inner_mask = (
                self.causal_mask[:cl, :cl] * self.decay_mask[:cl, :cl]
            ).repeat(bs, 1, 1)
            cross_mask = (
                torch.tensor(
                    [
                        [self.alpha ** (i + 1) for _ in range(d)]
                        for i in range(q.shape[1])
                    ]
                )
                .repeat(bs, 1, 1)
                .to(x.device)
            )

            # Retention update
            new_state = (
                k.transpose(-1, -2) @ (v * retention_mask)
                + self.alpha**chunk_size * state
            )

            # Output
            inner_chunk = (q @ k.transpose(-1, -2) / self.dim_sqrt * inner_mask) @ v
            cross_chunk = (q @ state) / self.dim_sqrt * cross_mask

            out = inner_chunk + cross_chunk
            outputs.append(out)

            state = new_state

        return torch.cat(outputs, dim=1)

    def forward(self, x, state=None, mode=ViRModes.PARALLEL, chunk_size=None):
        if mode is None:
            mode = self.mode

        if mode == ViRModes.PARALLEL:
            return self.forward_parallel(x)
        elif mode == ViRModes.RECURRENT:
            return self.forward_recurrent(x, state)
        elif mode == ViRModes.CHUNKWISE:
            return self.forward_chunkwise(x, chunk_size)
        else:
            raise ValueError(f"Unknown mode {mode}")


class MultiHeadRetention(nn.Module):
    def __init__(
        self,
        heads,
        embed_dim,
        max_len,
        alpha=DEFAULT_ALPHA,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
    ):
        super(MultiHeadRetention, self).__init__()
        self.n_heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.mode = mode
        self.chunk_size = chunk_size

        assert (
            embed_dim % heads == 0
        ), "Embedding dimension must be divisible by the number of heads"

        self.heads = nn.ModuleList(
            [
                Retention(embed_dim // heads, max_len, alpha, chunk_size)
                for _ in range(heads)
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mode=None, chunk_size=None):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        out = torch.cat(
            [
                head(
                    x[:, :, i * self.head_dim : (i + 1) * self.head_dim],
                    mode=mode,
                    chunk_size=chunk_size,
                )
                for i, head in enumerate(self.heads)
            ],
            dim=-1,
        )
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
        self,
        heads,
        embed_dim,
        max_len,
        alpha=DEFAULT_ALPHA,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
        dropout=0.1,
    ):
        super(ViRBlock, self).__init__()
        self.mode = mode
        self.chunk_size = chunk_size

        self.ln1 = nn.LayerNorm(embed_dim)
        self.retention = MultiHeadRetention(
            heads, embed_dim, max_len, alpha, mode, chunk_size
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mode=None, chunk_size=None):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        x = (
            self.dropout1(self.retention(self.ln1(x), mode=mode, chunk_size=chunk_size))
            + x
        )
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
        chunk_size=20,
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
        self.chunk_size = chunk_size

        # Embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ViR blocks
        self.blocks = nn.ModuleList(
            [
                ViRBlock(heads, embed_dim, max_len, alpha, mode, chunk_size, dropout)
                for _ in range(depth)
            ]
        )

        # Head
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, out_dim)

    def set_compute_mode(self, mode):
        self.mode = mode

    def forward(self, x, mode=None, chunk_size=None):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        # Patch embedding, positional embedding, CLS token
        x = self.patch_embed(x).permute(0, 2, 3, 1).flatten(1, 2)
        bs, sl = x.shape[:2]
        x = x + self.pos_embed.repeat(bs, 1, 1)[:, :sl]
        x = torch.cat(
            (x, self.class_token.repeat(bs, 1, 1)), dim=1
        )  # Important: CLS token goes last

        # Blocks
        for block in self.blocks:
            x = block(x, mode=mode, chunk_size=chunk_size)

        # Head on the CLS token
        x = self.linear(self.ln(x[:, -1]))

        return x


if __name__ == "__main__":
    """Tests that parallel and recurrent modes give the same output for ViR"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(16, 3, 224, 224).to(device)
    model = ViR(depth=12, heads=3, embed_dim=192).eval().to(device)

    with torch.no_grad():
        model.set_compute_mode(ViRModes.PARALLEL)
        y1 = model(x)

        model.set_compute_mode(ViRModes.RECURRENT)
        y2 = model(x)

        model.set_compute_mode(ViRModes.CHUNKWISE)
        chunk_size = 20
        y3 = model(x, chunk_size=chunk_size)

        assert torch.allclose(
            y1, y2, atol=1e-6
        ), "Parallel and recurrent modes should give the same output"

        assert torch.allclose(
            y1, y3, atol=1e-6
        ), "Parallel and chunkwise modes should give the same output"
