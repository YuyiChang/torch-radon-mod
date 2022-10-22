# TorchRadon-Mod: Fast Differentiable Routines for Computed Tomography

TorchRadon is a PyTorch extension written in CUDA that implements differentiable routines
for solving computed tomography (CT) reconstruction problems.

The library is designed to help researchers working on CT problems to combine deep learning
and model-based approaches.

### What modified?

1. **Add your own [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus) in:**

```
torch-radon/build_tools/__init__.py
```

line 61:

```python
def build(compute_capabilities=(60, 61, 70, 75, 80, 86), verbose=False, cuda_home="/usr/local/cuda", cxx="g++"):
```



2. **Support torch >= 1.8.0:**

torch version >= 1.8.0 drop the use of *torch.rfft* and *torch.irfft*, so use fft in module *torch.fft*.

```
torch-radon/torch_radon/__init__.py
```

line 96:

```python
sino_fft = torch.fft.fft(padded_sinogram, norm='ortho')  # torch 1.12.0
sino_fft = torch.stack((sino_fft.real, sino_fft.imag), -1)  # yes, do it
```

line 103:

```python
filtered_sinogram = torch.fft.ifft(torch.complex(filtered_sino_fft[..., 0], filtered_sino_fft[..., 1]), norm='ortho')  # torch 1.12.0
```

line 108:

```python
return np.real(filtered_sinogram).to(dtype=sinogram.dtype)
```



3. **TODO**
