# WeightForge

WeightForge is a small Python/PyTorch prototype that maps normalized matrix coordinates to scalar values with a multilayer perceptron. Its functions return generated tensors; the repository does not demonstrate trained-weight reconstruction or a validated model-compression method. The project metadata labels the release `1.0.0` and classifies its development status as Beta.

## Implemented functionality

- `WeightForge.forge_weight(out_dim, in_dim)` generates a tensor of shape `(out_dim, in_dim)` from row/column coordinates. It processes large outputs in chunks according to `chunk_size`.
- The coordinate network supports `silu`, `relu`, `gelu`, `tanh`, `sigmoid`, `elu`, and `leaky_relu` activations.
- `forge_layer()` generates a dictionary of tensors from named `(out_dim, in_dim)` shapes. `forge_model()` inspects a PyTorch module and returns generated tensors for its `nn.Linear` weights and biases; it does not assign them to the supplied model.
- `WeightForgeWrapper`, `WeightForgeConfig`, and `create_weight_forge()` provide convenience configuration and construction APIs.
- `count_parameters()` counts parameters in the coordinate network. `compression_ratio()` reports the arithmetic ratio of a supplied parameter count to that count; it does not measure compressed storage.

## Requirements and use from a checkout

The project metadata requires Python 3.10 or newer, PyTorch 2.0 or newer, and NumPy 1.24 or newer. From the repository root, install the declared requirements:

```bash
python -m pip install -r requirements.txt
```

Then use the module from the checkout:

```python
from weight_forge import create_weight_forge

forge = create_weight_forge(hidden_size=64, num_layers=3, chunk_size=1024)
weights = forge.forge_weight(out_dim=8, in_dim=4)
print(weights.shape)  # torch.Size([8, 4])
```

The source file also has a demonstration entry point that can be run from the repository root with `python weight_forge.py`.

**Packaging caveat:** At the reviewed commit, building a wheel with `python -m pip wheel --no-deps --no-build-isolation .` produced a wheel containing distribution metadata and the license, but not `weight_forge.py`. Consequently, installing that wheel does not provide the importable module; the usage above is for a source checkout, not an installed package.

## Distribution names and limitations

The `distribution` option accepts `fourier`, `xavier`, `kaiming`, `normal`, and `uniform`, but these names should not be interpreted as implemented samplers for those statistical distributions. The implementation applies fixed scaling to the coordinate network's output for the Xavier, Kaiming, Normal, and Uniform options. The `fourier` option uses the network output without additional scaling; the code does not implement a Fourier-feature embedding.

Generated values come from a randomly initialized coordinate network and its coordinates, not from a fit to a target matrix. The repository contains no training or fitting procedure, trained-weight recovery, serialization/compression format, or controlled accuracy/reconstruction/compression benchmark. `forge_model()` only returns a name-to-tensor mapping; callers are responsible for any further use or assignment. Its compression ratio is parameter-count arithmetic and excludes storage overhead and encoding costs.

## Tests

On the reviewed checkout (`44142d60c73d23af79ec87286bad6ef0900e2e2f`), `python -m pytest -q` completed with **29 passed**. The configured GitHub Actions workflow defines test runs on Python 3.10, 3.11, and 3.12; a remote workflow result was not checked for this draft.

## License

The repository includes an MIT License in [`LICENSE`](LICENSE), copyright 2026 Kapil Dhinwa. See that file for the license terms.
