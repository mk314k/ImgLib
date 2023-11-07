#include "../include/ImageManipulation.h"

__device__ void rgb2yuvKernel(float r, float g, float b, float& y, float& u, float& v) {
    y = 0.299f * r + 0.587f * g + 0.114f * b;
    u = -0.14713f * r - 0.288862f * g + 0.436f * b;
    v = 0.615f * r - 0.51499f * g - 0.10001f * b;
}

__device__ void yuv2rgbKernel(float y, float u, float v, float& r, float& g, float& b) {
    r = y + 1.13983f * v;
    g = y - 0.39465f * u - 0.5806f * v;
    b = y + 2.03211f * u;
}

__global__ void brightnessKernel(float* data, int size, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] *= factor;
    }
}

__global__ void contrastKernel(float* data, int size, float factor, float midpoint) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = factor * (data[idx] - midpoint) + midpoint;
    }
}

__global__ void color2grayKernel(float* r, float* g, float* b, float* gray, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        gray[idx] = 0.299f * r[idx] + 0.587f * g[idx] + 0.114f * b[idx];
    }
}

__global__ void saturateKernel(float* r, float* g, float* b, int size, float k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        r[idx] = (r[idx] - 0.5f) * k + 0.5f;
        g[idx] = (g[idx] - 0.5f) * k + 0.5f;
        b[idx] = (b[idx] - 0.5f) * k + 0.5f;
    }
}

// Implement CUDA versions of your other functions similarly

Image brightness(const Image& im, float factor) {
    // Allocate GPU memory for the input image
    im.allocateDeviceMemory();
    im.copyToGPU();

    int size = im.len();
    int numBlocks = (size + 255) / 256;
    brightnessKernel<<<numBlocks, 256>>>(im.device_image_data, size, factor);
    im.copyFromGPU();

    return im;
}

// Implement other functions similarly
