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

## 📁 Project Structure

```
codeV2/
├── main.py                   # Main entry point with benchmarking logic
├── filters.py                # Optimized image filter implementations (5 filters)
├── worker.py                 # Worker function for parallel processing
├── utils.py                  # Utility functions for image discovery
├── performance_metrics.json  # Generated performance results
└── processed_images/         # Output directory for processed images
```

## 🔧 Requirements

- Python 3.8+
- Pillow (PIL) library
- NumPy (for vectorized operations)

### Installation

```bash
pip install Pillow numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Configuration

Edit the configuration variables in `main.py`:

```python
SOURCE_DIR = r"path/to/food-101/dataset"  # Path to image dataset
OUTPUT_DIR = "processed_images"            # Output directory
MAX_IMAGES = 200                           # Number of images to process
CHOSEN_FILTER = 'all'                      # Options: grayscale, blur, sobel, sharpen, brightness, all
```

### Running the Application

```bash
python main.py
```

The program will:

1. Discover images from the source directory
2. Run benchmarks with different core counts (1, 2, 4, 8, max cores)
3. Apply all 5 filters to each image using both parallel paradigms
4. Generate performance metrics and save to `performance_metrics.json`
5. Display a performance analysis table in the console

## 📊 Performance Metrics

The system calculates and reports:

| Metric         | Formula                | Description                                       |
| -------------- | ---------------------- | ------------------------------------------------- |
| **Speedup**    | `T₁ / Tₙ`              | Performance gain compared to sequential execution |
| **Efficiency** | `(Speedup / n) × 100%` | How effectively cores are utilized                |
| **Throughput** | `Images / Time`        | Images processed per second                       |

### Sample Performance Results

Based on 200 images with all 5 filters:

```
==========================================================================================
                              PERFORMANCE ANALYSIS REPORT
==========================================================================================
Cores  | Paradigm        | Time (s)   | Speedup  | Efficiency | Throughput (img/s)
------------------------------------------------------------------------------------------
1      | Multiprocessing | 5.1622     | 1.00     | 100.0%     | 38.74
1      | Conc. Futures   | 5.2647     | 1.00     | 100.0%     | 37.99
------------------------------------------------------------------------------------------
2      | Multiprocessing | 3.1667     | 1.63     | 81.5%      | 63.16
2      | Conc. Futures   | 2.9305     | 1.80     | 89.8%      | 68.25
------------------------------------------------------------------------------------------
4      | Multiprocessing | 1.8878     | 2.73     | 68.4%      | 105.94
4      | Conc. Futures   | 1.8866     | 2.79     | 69.8%      | 106.01
------------------------------------------------------------------------------------------
8      | Multiprocessing | 1.5213     | 3.39     | 42.4%      | 131.47
8      | Conc. Futures   | 1.3722     | 3.84     | 48.0%      | 145.75
------------------------------------------------------------------------------------------
16     | Multiprocessing | 1.3473     | 3.83     | 23.9%      | 148.44
16     | Conc. Futures   | 1.4555     | 3.62     | 22.6%      | 137.41
------------------------------------------------------------------------------------------
```

## 🔬 Technical Implementation

### Parallel Paradigm 1: multiprocessing.Pool

```python
with multiprocessing.Pool(processes=num_cores) as pool:
    pool.map(worker.process_image_task, tasks)
```

**Characteristics:**

- Uses `Pool.map()` for automatic work distribution
- Each process loads and applies filters independently
- Built-in load balancing across worker processes

### Parallel Paradigm 2: concurrent.futures.ProcessPoolExecutor

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    list(executor.map(worker.process_image_task, tasks))
```

**Characteristics:**

- High-level abstraction over process pools
- Provides Future objects for async result handling
- Cleaner API with context manager support

## 📈 Performance Analysis

### Scalability Observations

1. **Near-linear speedup** up to 2-4 cores:

   - 2 cores: 1.63-1.80x speedup (~82-90% efficiency)
   - 4 cores: 2.73-2.79x speedup (~68-70% efficiency)

2. **Diminishing returns** beyond 4 cores:

   - 8 cores: 3.39-3.84x speedup (~42-48% efficiency)
   - 16 cores: 3.62-3.83x speedup (~23-24% efficiency)

3. **Bottlenecks identified:**
   - I/O operations (reading/writing images to disk)
   - Process creation and management overhead
   - Memory bandwidth limitations

### Paradigm Comparison

Both paradigms show **similar performance characteristics**, validating that they use the same underlying process-based parallelism. The slight variations are due to:

- Different overhead in task scheduling
- Memory management strategies
- API abstraction layers

## 🌐 GCP Deployment

### Steps for Google Cloud Platform

1. **Create a Compute Engine instance:**

   ```bash
   gcloud compute instances create image-processor \
     --machine-type=n1-standard-8 \
     --zone=us-central1-a
   ```

2. **SSH into the instance:**

   ```bash
   gcloud compute ssh image-processor --zone=us-central1-a
   ```

3. **Install dependencies:**

   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install Pillow numpy
   ```

4. **Upload code and dataset:**

   ```bash
   gcloud compute scp --recurse ./codeV2 image-processor:~ --zone=us-central1-a
   ```

5. **Run benchmarks:**
   ```bash
   python3 main.py
   ```

### Recommended Instance Types

| Instance Type  | vCPUs | RAM   | Use Case                          |
| -------------- | ----- | ----- | --------------------------------- |
| n1-standard-4  | 4     | 15 GB | Testing scalability up to 4 cores |
| n1-standard-8  | 8     | 30 GB | Optimal price/performance ratio   |
| n1-standard-16 | 16    | 60 GB | Maximum parallelism testing       |

## 📝 Code Improvements (V2 vs V1)

### 1. Vectorized Grayscale Conversion

**Before:**

```python
for x in range(width):
    for y in range(height):
        r, g, b = src_pixels[x, y]
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
```

**After (Optimized):**

```python
arr = np.asarray(img, dtype=np.float32)
gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
```

**Result:** ~2x faster processing

### 2. Fixed Sobel Edge Detection

**Before:**

```python
magnitude = magnitude / magnitude.max() * 255  # Can cause division by zero!
```

**After (Robust):**

```python
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)  # Safe clipping
```

**Result:** No crashes on edge cases, preserves gradient strength

### 3. RGBA Support

**Added:** Proper handling of images with alpha channels in brightness filter

## 📝 Dataset

Uses the [Food-101 Dataset](https://www.kaggle.com/datasets/dansbecker/food-101) from Kaggle.

**Dataset Details:**

- 101 food categories
- 1,000 images per category
- Total: 101,000 images

**Note:** Create manageable subsets for testing (recommended: 100-500 images).

## 🎯 Assignment Deliverables Checklist

- ✅ Implementation of 5 image filters (grayscale, blur, sobel, sharpen, brightness)
- ✅ Two parallel paradigms (multiprocessing + concurrent.futures)
- ✅ Performance metrics (speedup, efficiency, throughput)
- ✅ Comprehensive code comments and docstrings
- ✅ README with project description
- ✅ JSON output for performance analysis
- ✅ GCP deployment instructions

## 👥 Authors

CST435 - Parallel and Cloud Computing Assignment 2

## 📄 License

This project is for educational purposes as part of CST435 coursework.

---

## 🚨 Troubleshooting

### Issue: "No module named 'numpy'"

**Solution:**

```bash
pip install numpy
```

### Issue: "No images found"

**Solution:** Check that `SOURCE_DIR` points to the correct dataset location

### Issue: "Permission denied" on GCP

**Solution:**

```bash
chmod +x main.py
```

### Issue: Low performance/speedup

**Possible causes:**

- Dataset stored on slow disk (HDD vs SSD)
- CPU throttling due to thermal limits
- Competing processes consuming CPU resources

---

**For questions or issues, refer to the course materials or contact the instructor.**
