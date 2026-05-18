#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"

void run_uma_overhead_bench(ggml_backend_t backend, size_t total_gb = 1) {
    const size_t tensor_size = 64 * 1024 * 1024; // 64MB per tensor
    const size_t tensors_per_layer = 8;
    const size_t num_tensors = (total_gb * 1024ULL * 1024ULL * 1024ULL) / tensor_size;
    const size_t num_layers = num_tensors / tensors_per_layer;

    std::cout << "Simulating " << num_layers << " layers (" << num_tensors << " tensors total, "
              << tensors_per_layer << " per layer)..." << std::endl;

    struct ggml_init_params params = {
        .mem_size = total_gb * 1024ULL * 1024ULL * 1024ULL + (512 * 1024 * 1024),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Tensors for weight upload simulation
    std::vector<struct ggml_tensor *> weights;
    for (size_t i = 0; i < num_tensors; ++i) {
        weights.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 4096));
    }

    // Tensor for compute workload (fixed input A)
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 4096);

    // Pre-build all graphs and result tensors to remove construction cost from the loop
    std::vector<struct ggml_cgraph *> graphs(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        struct ggml_tensor * B = weights[layer * tensors_per_layer];
        struct ggml_tensor * res = ggml_mul_mat(ctx, A, B);
        graphs[layer] = ggml_new_graph(ctx);
        ggml_build_forward_expand(graphs[layer], res);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        ggml_free(ctx);
        return;
    }

    std::vector<float> source_data(tensor_size / sizeof(float), 1.0f);

    const int num_iterations = 10;
    std::vector<double> all_layer_times;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; ++iter) {
        // 1. Prime the pump for each iteration: Upload ALL tensors for the first layer
        for (size_t t = 0; t < tensors_per_layer; ++t) {
            ggml_backend_tensor_set_async(backend, weights[t], source_data.data(), 0, tensor_size);
        }

        for (size_t layer = 0; layer < num_layers; ++layer) {
            auto layer_start = std::chrono::high_resolution_clock::now();

            // 2. Queue the upload for the NEXT layer's tensors (async)
            if (layer + 1 < num_layers) {
                for (size_t t = 0; t < tensors_per_layer; ++t) {
                    size_t idx = (layer + 1) * tensors_per_layer + t;
                    ggml_backend_tensor_set_async(backend, weights[idx], source_data.data(), 0, tensor_size);
                }
            }

            // 3. Compute on the layer using the pre-built graph
            ggml_backend_graph_compute(backend, graphs[layer]);

            // 4. Sync - this waits for the compute to finish (and the flush of the next upload)
            ggml_backend_synchronize(backend);

            auto layer_end = std::chrono::high_resolution_clock::now();
            all_layer_times.push_back(std::chrono::duration<double, std::milli>(layer_end - layer_start).count());
        }
    }
    auto end_total = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    double sum = 0;
    for (double t : all_layer_times) sum += t;
    double avg_layer_ms = sum / all_layer_times.size();

    std::cout << "Total wall time: " << total_ms << " ms" << std::endl;
    std::cout << "Avg per-layer latency (upload + compute): " << avg_layer_ms << " ms" << std::endl;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    bool force_transfer_queue = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--force-transfer-queue") == 0) {
            force_transfer_queue = true;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 2;
        }
    }

    if (force_transfer_queue) {
        setenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "1", 1);
        unsetenv("GGML_VK_DISABLE_TRANSFER_QUEUE");
    } else {
        unsetenv("GGML_VK_DISABLE_TRANSFER_QUEUE");
        unsetenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE");
    }

    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) return 1;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    std::cout << "Device: " << ggml_backend_dev_name(dev) << std::endl;

    if (force_transfer_queue) std::cout << "Mode: Transfer Queue FORCED ENABLED" << std::endl;
    else std::cout << "Mode: Transfer Queue DEFAULT" << std::endl;

    run_uma_overhead_bench(backend, 8);

    ggml_backend_free(backend);
    return 0;
}
