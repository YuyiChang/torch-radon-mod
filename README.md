# [TorchRadon](https://github.com/matteo-ronchetti/torch-radon)-Mod: Fast Differentiable Routines for Computed Tomography

[TorchRadon](https://github.com/matteo-ronchetti/torch-radon) is a PyTorch extension written in CUDA that implements differentiable routines for solving computed tomography (CT) reconstruction problems.

The library is designed to help researchers working on CT problems to combine deep learning and model-based approaches.

<h2 align="center" style="color: black">
  Installation
</h2>

- On Linux with PyTorch >= 1.8, CUDA and GCC, run

  ```sh
  $ git clone git@github.com:CandleHouse/torch-radon-mod.git --depth 1
  $ cd torch-radon-mod
  $ export CUDA_HOME="/usr/local/cuda" # Specify this according to your situation.
  $ python setup.py install
  ```

- Run `examples/fbp.py` to check whether it is successfully installed.

<h2 align="center" style="color: blue">
  What's different?
</h2>


1. **Support CUDA GPUs with different [Compute Capability](https://developer.nvidia.com/cuda-gpus)**.

   In `setup.py`, modify Line 11 if you need.
   
   ```py
   build(compute_capabilities=(60, 61, 70, 75, 80, 86), cuda_home=cuda_home)
   ```


2. **Support torch >= 1.8.0.**

   torch >= 1.8.0 deprecated *torch.rfft* and *torch.irfft*. We should use fft in module *torch.fft* as substations. Refer to the [modifications here](https://github.com/CandleHouse/torch-radon-mod/commit/4fff5999c788829da9b044e3d22db1c650043ead#diff-d039beda73403cb00f204b3df845779f304448852484347ce8b18b0449d88b1f).

3. **Remove codes that we don't use.**

   e.g. alpha-shearlet transform, benchmarks, etc.
