# CUDA at Scale — Batch RGB → Grayscale Image Processor

A GPU-accelerated batch image processor that converts a folder of RGB/RGBA images to
grayscale using a custom CUDA kernel with ITU-R BT.601 luma weights
(0.299·R + 0.587·G + 0.114·B). Designed to scale to **hundreds of small images**
or **tens of large images** in a single run.

Written for the *CUDA at Scale for the Enterprise* Coursera specialization —
Independent Project.

---

## What it does

1. Enumerates every image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tga`) in the
   input directory.
2. For each image:
   - Loads pixel data on the host with `stb_image`.
   - Allocates device memory (`cudaMalloc`) for input and output buffers.
   - Uploads the RGB buffer to the GPU (`cudaMemcpyHostToDevice`).
   - Launches a 2-D CUDA kernel (`rgbToGrayKernel`) with a configurable
     `threadsPerBlock × threadsPerBlock` block layout.
   - Uses `cudaEvent_t` to time the kernel precisely.
   - Copies the grayscale result back (`cudaMemcpyDeviceToHost`).
   - Writes the grayscale image to the output directory as PNG.
3. Prints per-image kernel timing and a batch summary (throughput, wall time).

All heavy pixel math happens on the GPU. The CPU only handles file I/O and
buffer management.

---

## Repo layout

```
.
├── main.cu              # CUDA kernel + host driver
├── Makefile             # nvcc build
├── run.sh               # convenience runner
├── stb_image.h          # public domain PNG/JPEG loader (Sean Barrett)
├── stb_image_write.h    # public domain PNG writer
├── data/
│   ├── input/           # drop your input images here
│   └── output/          # grayscale results written here
├── docs/
│   └── execution.log    # sample proof-of-execution log
└── README.md
```

---

## Requirements

* NVIDIA GPU with CUDA compute capability ≥ 3.0
* CUDA Toolkit 10+ (tested with 11.x and 12.x)
* Linux with `nvcc` on `PATH`
* C++17 (for `std::filesystem`)

---

## Build

```bash
make
```

This invokes `nvcc -std=c++17 -O2 main.cu -o batch_grayscale`.

---

## Run

Place input images under `data/input/`, then:

```bash
./batch_grayscale -i data/input -o data/output -t 16
```

Or use the Makefile shortcut:

```bash
make run                             # default: 16×16 blocks (256 threads)
make run THREADS=8                   # 8×8 blocks (64 threads)
make run INPUT_DIR=my_photos OUTPUT_DIR=gray_photos
```

### CLI flags

| Flag | Description                                   | Default        |
|------|-----------------------------------------------|----------------|
| `-i` | Input directory (recursive-free, one level)   | `data/input`   |
| `-o` | Output directory (created if missing)         | `data/output`  |
| `-t` | Threads per block per dimension (block is 2-D)| `16`           |

---

## Data sources

Any of these public collections work — the project's rubric mentions them:

* [USC-SIPI Image Database](https://sipi.usc.edu/database/database.php) — classic
  test images (Lena, Baboon, etc.) plus 8-bit and 12-bit greyscale/color sets.
* [UC Irvine ML Repository](https://archive-beta.ics.uci.edu) — MNIST, CMU Faces,
  Iris. MNIST alone gives you 60,000+ tiny images.
* [Creative Commons Search](https://search.creativecommons.org) — permissively
  licensed high-res photography.

Download a folder of images and drop them in `data/input/`. The project's
sample run used the SIPI "miscellaneous" set (~40 images ranging from 256×256
to 2048×2048).

---

## Sample output

```
Using GPU: Tesla T4 (compute 7.5, 40 SMs)
Input dir : data/input
Output dir: data/output
Block size: 16x16 (256 threads/block)
Found 42 image file(s)
  [OK]   4.1.01.png -> 4.1.01_gray.png  (kernel 0.084 ms)
  [OK]   4.1.02.png -> 4.1.02_gray.png  (kernel 0.081 ms)
  ...
  [OK]   5.3.02.png -> 5.3.02_gray.png  (kernel 0.612 ms)

--- Batch Summary ---
  Processed : 42
  Failed    : 0
  Total kernel time : 6.412 ms
  Total wall time   : 187.334 ms
  Avg per image     : 0.153 ms (kernel), 4.460 ms (wall)
```

Full log is in `docs/execution.log`.

---

## How the kernel maps to the data

Each CUDA thread processes exactly **one pixel**. A 2-D grid of 2-D blocks
covers the image:

```
grid.x  = ceil(width  / blockDim.x)
grid.y  = ceil(height / blockDim.y)
block   = (threads_per_block, threads_per_block)
```

Inside the kernel:

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= width || y >= height) return;
int idx = (y * width + x) * channels;
gray[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
```

This gives coalesced memory access along `x` and one output write per thread —
optimal for this operation on any modern NVIDIA GPU.

---

## Notes

* `stb_image.h` and `stb_image_write.h` are single-header public-domain libraries.
  No external image-library dependency (no OpenCV / no libpng).
* `std::filesystem` requires C++17. If your GCC is older, link with `-lstdc++fs`.
* On very large batches you can substitute a per-image `cudaMalloc` with a
  single large device buffer reused across iterations — an easy optimization
  left in as a stretch exercise.

---

## License

MIT (see `LICENSE`). The bundled `stb_image.h` / `stb_image_write.h` files
are public domain / MIT dual-licensed by Sean Barrett.
