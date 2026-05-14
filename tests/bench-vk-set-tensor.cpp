#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

void benchmark_size(ggml_backend_t backend, struct ggml_context * ctx, size_t size_bytes) {
    int64_t nelements = size_bytes / sizeof(float);
    if (nelements == 0) nelements = 1;
    size_t actual_size = nelements * sizeof(float);

    // Allocate buffer on backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, actual_size);
    if (!buffer) {
        std::cerr << "Failed to allocate buffer of size " << size_bytes << " bytes" << std::endl;
        return;
    }

    // Create tensor metadata
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);
    if (!tensor) {
        std::cerr << "Failed to create tensor metadata" << std::endl;
        ggml_backend_buffer_free(buffer);
        return;
    }

    // Link tensor to buffer
    tensor->buffer = buffer;
    if (ggml_backend_buffer_init_tensor(buffer, tensor) != GGML_STATUS_SUCCESS) {
        std::cerr << "Failed to init tensor with buffer" << std::endl;
        ggml_backend_buffer_free(buffer);
        return;
    }

    // Satisfy the assert in ggml_backend_tensor_set_async
    tensor->data = (void*)0x1000;

    std::vector<float> data(nelements, 1.0f);

    // Warmup
    ggml_backend_tensor_set_2d_async(backend, tensor, data.data(), 0, sizeof(float), 1, sizeof(float), sizeof(float));
    ggml_backend_synchronize(backend);

    const int iterations = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_tensor_set_2d_async(backend, tensor, data.data(), 0, sizeof(float), 1, sizeof(float), sizeof(float));
        ggml_backend_synchronize(backend);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double avg_ms = duration.count() / iterations;

    std::cout << std::setw(10) << size_bytes / 1024 << " KB"
              << std::setw(15) << std::fixed << std::setprecision(4) << avg_ms << " ms" << std::endl;

    ggml_backend_buffer_free(buffer);
}

int main() {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) {
        std::cerr << "Failed to initialize Vulkan backend" << std::endl;
        return 1;
    }

    std::cout << "Benchmarking ggml_backend_vk_set_tensor_2d_async" << std::endl;
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Avg Time" << std::endl;
    std::cout << std::string(25, '-') << std::endl;

    // Sizes from 2KB up to 1GB, doubling each time
    for (size_t size = 2 * 1024; size <= 1024 * 1024 * 1024; size *= 2) {
        benchmark_size(backend, ctx, size);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
