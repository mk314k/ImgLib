__global__ void addKernel(float* im1, float* im2, float* output, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
      output[index] = im1[index] + im2[index];
    }
  }
  
  // Member function to invoke the CUDA addition kernel
  void Image::processWithCUDA() {
    // Check if CUDA is available and data is on the GPU
    if (cudaGetDeviceCount(nullptr) == 0 || device_image_data == nullptr) {
      // Handle error or fallback to CPU processing
      return;
    }
  
    int size = len();
    const int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
  
    float* d_im1 = device_image_data;  // Assuming im1 is already on the GPU
    float* d_im2 = device_image_data;  // Assuming im2 is already on the GPU
    float* d_output;
  
    // Allocate GPU memory for the output
    cudaMalloc((void**)&d_output, size * sizeof(float));
  
    // Launch the kernel
    addKernel<<<numBlocks, blockSize>>>(d_im1, d_im2, d_output, size);
  
    // Copy the result back to the host (if needed)
    cudaMemcpy(image_data.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  
    // Free GPU memory
    cudaFree(d_output);
  }
  
  // Member function to allocate GPU memory
  void Image::allocateDeviceMemory() {
    int size = len();
    cudaMalloc((void**)&device_image_data, size * sizeof(float));
  }
  
  // Member function to deallocate GPU memory
  void Image::deallocateDeviceMemory() {
    if (device_image_data) {
      cudaFree(device_image_data);
      device_image_data = nullptr;
    }
  }