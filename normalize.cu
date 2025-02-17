extern "C"
__global__ void normalize_image(float* img, int width, int height, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = ((b * height + y) * width + x) * 3; // Updated for batch processing

    if (x < width && y < height && b < batch_size) {
        img[idx] /= 255.0;
        img[idx + 1] /= 255.0;
        img[idx + 2] /= 255.0;
    }
}
