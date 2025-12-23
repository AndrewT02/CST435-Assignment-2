# Parallel Image Processing System

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

### 1. Create the VM Instance
- Machine Type: e2-standard-8 (or any instance with 8+ vCPUs).
- Boot Disk: Ubuntu 22.04 LTS x86/64.
- Size: 128 GB.
- Access: Once running, click SSH to open the terminal.

### 2. Setup System 
Run these commands to update the system and install the python:
- run `sudo apt update`
- run `sudo apt install python-is-python3 python3-pip unzip git -y`

### 3. Clone Repository & Install Dependencies
- run `git clone https://github.com/AndrewT02/CST435-Assignment-2`
- run `cd CST435-Assignment-2`
- run `python -m pip install -r requirements.txt`

### 4. Download Dataset
- run `python dataset.py`
- Important: Look at the output of this command. It will print a directory path where the data was saved. Copy this path.
- Update main.py with the new path:
  - nano main.py
    - Use the arrow keys to find the line: SOURCE_DIR = "..."
    - Delete the existing path inside the quotes.
    - Paste the path you copied from step 1.
    - Save & Exit: Press CTRL+O, Enter, then CTRL+X.   

### 5. Run the Application
- run `python main.py`

### 6. Download Results
- run `readlink -f performance_metrics.json`
- Select Download file.
- Paste the path you just copied.
- Click Download.
