# ViR: Vision Retention Networks (unofficial re-implementation)

Unofficial re-implementation of [*ViR: Vision Retention Networks*](https://arxiv.org/abs/2310.19731) by *Ali Hatamizadeh, Michael Ranzinger, Jan Kautz*.

## Usage
```python
from vir import ViRm, ViRModes

model = ViR(
  out_dim=10,
  patch_size=14,
  depth=12,
  heads=12,
  embed_dim=768,
  max_len=257,
)

x = torch.randn(16, 257, 768)

# All forward modes (parallel, recurrent, chunkwise) give the same output
# Parallel
y_parallel = model(x, ViRModes.PARALLEL)

# Recurrent
y_recurrent = model(x, ViRModes.RECURRENT)

# Parallel
y_chunkwise = model(x, mode=ViRModes.CHUNKWISE, chunk_size=20)
```

## Classification performance on ImageNette
A Vision Retention Network tiny (3 heads, 12 layers, 192 embed dim) achieves a 100% accuracy on the [Imagenette](https://huggingface.co/datasets/frgfm/imagenette) dataset after roughly 40 epochs with a batch size of 64.

## Citation
If you find this code useful for your research, please cite the repo:

```bibtex
@software{Pulfer_ViR_2023,
author = {Pulfer, Brian},
month = November
title = {{Vision Retention Networks (unofficial re-implementation)}},
url = {https://github.com/BrianPulfer/vision-retention-networks},
year = {2023}
}
```

## License
The code is released with the Apache 2.0 [license](LICENSE).
