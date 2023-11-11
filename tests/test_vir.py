import torch
from vir import ViRModes, ViR


def test_parallel_recurrent_same():
    """Tests that parallel and recurrent modes give the same output for ViR"""
    x = torch.randn(16, 3, 224, 224).cuda(0)
    model = ViR(depth=12, heads=3, embed_dim=192).eval().cuda(0)

    with torch.no_grad():
        model.set_compute_mode(ViRModes.PARALLEL)
        y1 = model(x)

        model.set_compute_mode(ViRModes.RECURRENT)
        y2 = model(x)

        assert torch.allclose(
            y1, y2, atol=1e-6
        ), "Parallel and recurrent modes should give the same output"
