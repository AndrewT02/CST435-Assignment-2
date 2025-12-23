# Parallel Image Processing System (Optimized Version)

A high-performance parallel image processing pipeline that applies multiple filters to images from the Food-101 dataset using Python's parallel computing paradigms with **optimized vectorized operations**.

## 📋 Project Overview

This project implements an optimized image processing system that demonstrates parallel computing concepts by applying five different filters to a collection of images. The implementation compares two Python parallel paradigms:

1. **multiprocessing.Pool** - Low-level process pool implementation
2. **concurrent.futures.ProcessPoolExecutor** - High-level executor-based parallelism

### ✨ Key Optimizations

- **Vectorized grayscale conversion** using NumPy for ~2x performance boost
- **Optimized Sobel edge detection** with proper gradient magnitude calculation
- **RGBA support** in brightness adjustment filter
- **Robust error handling** to prevent division by zero issues

## 🎨 Image Filters Implemented

| Filter                   | Description                | Implementation                                             |
| ------------------------ | -------------------------- | ---------------------------------------------------------- |
| **Grayscale**            | Converts RGB to grayscale  | Vectorized luminance formula: `0.299R + 0.587G + 0.114B`   |
| **Gaussian Blur**        | 3×3 smoothing kernel       | Convolution with normalized Gaussian kernel                |
| **Sobel Edge Detection** | Detects edges              | Gradient magnitude: `sqrt(Gx² + Gy²)` with proper clipping |
| **Sharpening**           | Enhances edges and details | Laplacian-based sharpening kernel                          |
| **Brightness**           | Adjusts image brightness   | Pixel-wise multiplication with RGBA support                |

## Running in GCP

### Steps in GCP 
- VM must have at least 8 cores and sufficient memory
- Boot  Disk: Ubuntu 22.04 LTS X86/64
- Size: 128Gb
- Once created, open SSH

### Running python code 
- Run the following code 
  - sudo apt update
  - sudo apt install python3-pip unzip -y
- Upload all related files:
  -   filters.py
  -   main.py
  -   utils.py
  -   worker.py
  -   dataset.py
  -   requirements.txt
- Once uploaded, run pip install -r requirements.txt
- Then run python dataset.py
- The directory of where the dataset is save should be printed.
- Copy the directory, run nano main.py then replace the directory with the old one.
- Once completed, press CTRL+O then press "Enter" then CRTL+X
- Once completed, run python main.py
- Then run ls performance_metrics.json
- Copy the directory location and click "Download" and paste the directory in to download the json file.
