from time import time
from argparse import ArgumentParser

import torch

from vir import ViR, ViRModes


def measure(fn, *args, **kwargs):
    start = time()
    fn(*args, **kwargs)
    end = time()
    return end - start


def main(args):
    """Benchmark parallel, recurrent and chunkwise modes. Show memory usage and runtime. Try multiple image resolutions (i.e. sequence lengths)."""
    # Unpacking arguments
    alpha = 1.00
    patch_size = args["patch_size"]
    depth = args["depth"]
    heads = args["heads"]
    embed_dim = args["embed_dim"]
    chunk_size = args["chunk_size"]
    batch_size = args["batch_size"]
    image_sizes = [224 * i for i in range(1, 2)]
    # image_sizes = [224 * i for i in range(1, 4)]

    # Creating model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = (image_sizes[-1] // patch_size) ** 2 + 1
    model = (
        ViR(
            out_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            embed_dim=embed_dim,
            max_len=max_len,
            alpha=alpha,
            chunk_size=chunk_size,
        )
        .eval()
        .to(device)
    )

    for image_size in image_sizes:
        x = torch.randn(batch_size, 3, image_size, image_size).to(device)

        with torch.no_grad():
            model.set_compute_mode(ViRModes.PARALLEL)
            time_parallel = measure(model, x)

            model.set_compute_mode(ViRModes.CHUNKWISE)
            time_chunkwise = measure(model, x, chunk_size=chunk_size)

            print(f"\nImage size: {image_size}")
            print(f"\tParallel: {time_parallel:.3f} s")
            print(f"\tChunkwise: {time_chunkwise:.3f} s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = vars(parser.parse_args())
    print("\nRunning benchmark with the following parameters:\n", args, "\n")
    main(args)
