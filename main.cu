/*
 * CUDA at Scale Independent Project — Batch RGB → Grayscale Image Processor
 *
 * Processes a folder of images (100+ small OR 10+ large) end-to-end on the GPU:
 *   1. Read input files from a directory (STB image loader — bundled)
 *   2. Upload pixel buffers to device memory
 *   3. Launch a CUDA kernel that converts RGB → luminance grayscale
 *      using the ITU-R BT.601 coefficients (0.299R + 0.587G + 0.114B)
 *   4. Copy results back to host
 *   5. Write grayscale outputs to output directory as PNG
 *
 * Usage:
 *   ./batch_grayscale -i <input_dir> -o <output_dir> [-t <threads_per_block>]
 *
 * Example:
 *   ./batch_grayscale -i data/input -o data/output -t 256
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernel: RGB (or RGBA) → grayscale using ITU-R BT.601 luma weights
// ---------------------------------------------------------------------------
__global__ void rgbToGrayKernel(const unsigned char *rgb,
                                unsigned char *gray,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    unsigned char r = rgb[idx + 0];
    unsigned char g = rgb[idx + 1];
    unsigned char b = rgb[idx + 2];
    // ITU-R BT.601 luma
    float luma = 0.299f * r + 0.587f * g + 0.114f * b;
    gray[y * width + x] = static_cast<unsigned char>(luma);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Args {
    std::string input_dir  = "data/input";
    std::string output_dir = "data/output";
    int threads_per_block  = 16; // 16x16 block = 256 threads
};

static Args parseArgs(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-i" && i + 1 < argc)      a.input_dir  = argv[++i];
        else if (k == "-o" && i + 1 < argc) a.output_dir = argv[++i];
        else if (k == "-t" && i + 1 < argc) a.threads_per_block = std::atoi(argv[++i]);
        else if (k == "-h" || k == "--help") {
            printf("Usage: %s -i <input_dir> -o <output_dir> [-t <threads_per_block>]\n",
                   argv[0]);
            exit(0);
        }
    }
    return a;
}

static bool isImageFile(const fs::path &p) {
    std::string ext = p.extension().string();
    for (auto &c : ext) c = std::tolower(c);
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".tga";
}

// Process one image on the GPU; returns elapsed kernel time in ms
static float processImage(const fs::path &in_path, const fs::path &out_path,
                          int threads_per_block) {
    int w, h, ch;
    unsigned char *h_rgb = stbi_load(in_path.string().c_str(), &w, &h, &ch, 0);
    if (!h_rgb) {
        fprintf(stderr, "  ! failed to load %s\n", in_path.c_str());
        return -1.0f;
    }
    if (ch < 3) {
        fprintf(stderr, "  ! %s has %d channels, need >=3, skipping\n",
                in_path.c_str(), ch);
        stbi_image_free(h_rgb);
        return -1.0f;
    }

    size_t rgb_size  = static_cast<size_t>(w) * h * ch;
    size_t gray_size = static_cast<size_t>(w) * h;

    unsigned char *d_rgb = nullptr, *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb,  rgb_size));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_size));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice));

    dim3 block(threads_per_block, threads_per_block);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    rgbToGrayKernel<<<grid, block>>>(d_rgb, d_gray, w, h, ch);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<unsigned char> h_gray(gray_size);
    CUDA_CHECK(cudaMemcpy(h_gray.data(), d_gray, gray_size,
                          cudaMemcpyDeviceToHost));

    fs::create_directories(out_path.parent_path());
    stbi_write_png(out_path.string().c_str(), w, h, 1, h_gray.data(), w);

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    stbi_image_free(h_rgb);
    return ms;
}

int main(int argc, char **argv) {
    Args a = parseArgs(argc, argv);

    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("Using GPU: %s (compute %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Input dir : %s\n", a.input_dir.c_str());
    printf("Output dir: %s\n", a.output_dir.c_str());
    printf("Block size: %dx%d (%d threads/block)\n",
           a.threads_per_block, a.threads_per_block,
           a.threads_per_block * a.threads_per_block);

    if (!fs::exists(a.input_dir)) {
        fprintf(stderr, "! input dir does not exist: %s\n", a.input_dir.c_str());
        return EXIT_FAILURE;
    }
    fs::create_directories(a.output_dir);

    std::vector<fs::path> images;
    for (auto &entry : fs::directory_iterator(a.input_dir))
        if (entry.is_regular_file() && isImageFile(entry.path()))
            images.push_back(entry.path());

    printf("Found %zu image file(s)\n", images.size());
    if (images.empty()) return EXIT_SUCCESS;

    auto t0 = std::chrono::steady_clock::now();
    float total_kernel_ms = 0.0f;
    int ok = 0, fail = 0;

    for (auto &p : images) {
        fs::path out = fs::path(a.output_dir) / (p.stem().string() + "_gray.png");
        float ms = processImage(p, out, a.threads_per_block);
        if (ms >= 0) {
            total_kernel_ms += ms;
            ++ok;
            printf("  [OK]   %s -> %s  (kernel %.3f ms)\n",
                   p.filename().c_str(), out.filename().c_str(), ms);
        } else {
            ++fail;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n--- Batch Summary ---\n");
    printf("  Processed : %d\n", ok);
    printf("  Failed    : %d\n", fail);
    printf("  Total kernel time : %.3f ms\n", total_kernel_ms);
    printf("  Total wall time   : %.3f ms\n", wall_ms);
    if (ok > 0)
        printf("  Avg per image     : %.3f ms (kernel), %.3f ms (wall)\n",
               total_kernel_ms / ok, wall_ms / ok);
    return EXIT_SUCCESS;
}
