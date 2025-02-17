extern "C"
__global__ void tensor_to_image(float* tensor, unsigned char* image, int width, int height, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = ((b * height + y) * width + x) * 3;  // Combined index for the entire batch
    int img_idx = (b * height * width + y * width + x) * 3;  // Index for the individual image within the batch

    if (x < width && y < height && b < batch_size) {
        // Normalize and convert to image
        image[img_idx] = (unsigned char)(tensor[idx] * 255.0);
        image[img_idx + 1] = (unsigned char)(tensor[idx + 1] * 255.0);
        image[img_idx + 2] = (unsigned char)(tensor[idx + 2] * 255.0);
    }
}