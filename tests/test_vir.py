import torch
from vir import ViRModes, ViR


def test_parallel_recurrent_same():
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
