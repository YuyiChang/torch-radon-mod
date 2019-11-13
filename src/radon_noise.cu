#include "radon_noise.h"

__global__ void initialize_random_states(curandState *state, const uint seed){
    const uint sequence_id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, sequence_id, 0, &state[sequence_id]);
}

__global__ void radon_sinogram_noise(float* sinogram, curandState *state, const float signal, const uint width, const uint height){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    // load curand state in local memory
    curandState localState = state[tid];

    // loop down the sinogram adding noise
    for(uint yy = y; yy < height; yy += y_step){
        uint pos = yy * width + x;
        // measured signal = signal * exp(-sinogram[pos])
        // then apply poisson noise
        float reading = curand_poisson(&localState, signal * exp(-sinogram[pos]));
        // convert back to sinogram scale
        sinogram[pos] = -log(reading / signal);
    }

    // save curand state back in global memory
    state[tid] = localState;
}

RadonNoiseGenerator::RadonNoiseGenerator(const uint seed){
    // allocate random states
    checkCudaErrors(cudaMalloc((void **)&states, 4096 * sizeof(curandState)));

    this->set_seed(seed);
}

void RadonNoiseGenerator::set_seed(const uint seed){
    initialize_random_states<<<32,128>>>(states, seed);
}

void RadonNoiseGenerator::add_noise(float* sinogram, const float signal, const uint width, const uint height){
    // TODO consider case width > 1024
    radon_sinogram_noise<<<dim3(1, 4), dim3(width, 1024/width)>>>(sinogram, states, signal, width, height);
}

void RadonNoiseGenerator::free(){
    if(this->states != nullptr){
        checkCudaErrors(cudaFree(this->states));
    }
}
