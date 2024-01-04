import torch
import torch.nn as nn


class ViRModes:
    PARALLEL = "parallel"
    RECURRENT = "recurrent"
    CHUNKWISE = "chunkwise"


class Retention(nn.Module):
    def __init__(
        self,
        embed_dim,
        max_len,
        alpha,
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

        indices = torch.arange(max_len).reshape(1, -1)
        self.register_buffer(
            "decay_mask",
            (alpha ** (indices.t() - indices)).tril(),
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
        state = torch.zeros(batch_size, dim, dim, device=x.device)
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

        # Adding dummy tokens to make the sequence length divisible by chunk_size
        if sl % chunk_size != 0:
            x = torch.cat(
                [x, torch.zeros(bs, chunk_size - sl % chunk_size, d, device=x.device)],
                dim=1,
            )
        n_chunks = x.shape[1] // chunk_size

        # Running all chunks in parallel
        x = x.reshape(bs, n_chunks, chunk_size, d)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        M = (
            self.causal_mask[:chunk_size, :chunk_size]
            * self.decay_mask[:chunk_size, :chunk_size]
        ).repeat(bs, n_chunks, 1, 1)

        inner_chunk = (q @ k.transpose(-1, -2) / self.dim_sqrt * M) @ v

        # Updating outputs with chunk-wise recurrent
        retention_mask = (
            torch.tensor(
                [self.alpha ** (chunk_size - i - 1) for i in range(chunk_size)],
                device=x.device,
            )
            .repeat(bs, d, 1)
            .transpose(-1, -2)
        )

        cross_mask = (
            torch.tensor(
                [self.alpha ** (i + 1) for i in range(chunk_size)], device=x.device
            )
            .repeat(bs, n_chunks, d, 1)
            .transpose(-1, -2)
        )

        states = torch.zeros(bs, n_chunks, d, d, device=x.device)
        for i in range(1, n_chunks):
            chunk_state = k[:, i - 1].transpose(-1, -2) @ (v[:, i - 1] * retention_mask)
            states[:, i] = chunk_state + states[:, i - 1] * self.alpha**chunk_size

        cross_chunk = (q @ states) / self.dim_sqrt * cross_mask

        # Combining inner and cross chunk
        out = inner_chunk + cross_chunk

        # Removing dummy tokens
        out = out.flatten(1, 2)[:, :sl]
        return out

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
        alphas=None,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
    ):
        super(MultiHeadRetention, self).__init__()
        self.n_heads = heads
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.alphas = alphas
        self.head_dim = embed_dim // heads
        self.mode = mode
        self.chunk_size = chunk_size

        if alphas is None:
            alphas = [1 - 2 ** (-5 - i) for i in range(heads)]

        assert len(alphas) == heads, "Number of alphas must match number of heads"

        assert (
            embed_dim % heads == 0
        ), "Embedding dimension must be divisible by the number of heads"

        self.heads = nn.ModuleList(
            [
                Retention(embed_dim // heads, max_len, alpha, mode, chunk_size)
                for alpha in alphas
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
        alphas=None,
        mode=ViRModes.PARALLEL,
        chunk_size=20,
        dropout=0.1,
    ):
        super(ViRBlock, self).__init__()
        self.mode = mode
        self.chunk_size = chunk_size

        self.ln1 = nn.LayerNorm(embed_dim)
        self.retention = MultiHeadRetention(
            heads, embed_dim, max_len, alphas, mode, chunk_size
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
        patch_size=14,
        depth=12,
        heads=12,
        embed_dim=768,
        max_len=256,
        alphas=None,
        mode=ViRModes.CHUNKWISE,
        chunk_size=256,
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
        self.alphas = alphas
        self.mode = mode
        self.chunk_size = chunk_size

        # Embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

        # ViR blocks
        self.blocks = nn.ModuleList(
            [
                ViRBlock(heads, embed_dim, max_len, alphas, mode, chunk_size, dropout)
                for _ in range(depth)
            ]
        )

        # Head
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, mode=None, chunk_size=None, reshape=False):
        if mode is None:
            mode = self.mode

        if chunk_size is None:
            chunk_size = self.chunk_size

        # Patch embedding, positional embedding
        x = self.patch_embed(x).permute(0, 2, 3, 1).flatten(1, 2)
        bs, sl = x.shape[:2]
        x = x + self.pos_embed.repeat(bs, 1, 1)[:, :sl]

        # Blocks
        for block in self.blocks:
            x = block(x, mode=mode, chunk_size=chunk_size)

        # Layer Norm
        x = self.ln(x)

        # Reshape
        if reshape:
            ps = int(x.shape[1] ** 0.5)
            x = x.reshape(bs, ps, ps, self.embed_dim).permute(0, 3, 1, 2)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(16, 3, 224, 224).to(device)
    model = ViR(depth=12, heads=3, embed_dim=192).eval().to(device)

    with torch.no_grad():
        y1 = model(x, mode=ViRModes.PARALLEL)
        y2 = model(x, mode=ViRModes.RECURRENT)
        y3 = model(x, mode=ViRModes.CHUNKWISE, chunk_size=20)

        assert torch.allclose(
            y1, y2, atol=1e-5
        ), "Parallel and recurrent modes should give the same output"

        assert torch.allclose(
            y1, y3, atol=1e-5
        ), "Parallel and chunkwise modes should give the same output"
