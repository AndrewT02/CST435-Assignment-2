"""
filters.py - Image Filter Implementations

This module contains implementations of five image processing filters:
1. Grayscale Conversion - Luminance-based RGB to grayscale
2. Gaussian Blur - 3x3 smoothing convolution
3. Sobel Edge Detection - Gradient-based edge detection
4. Image Sharpening - Laplacian-based edge enhancement
5. Brightness Adjustment - Pixel-wise intensity scaling

All filters operate on individual pixels or small neighborhoods,
making them suitable for parallel processing.

Author: CST435 Assignment 2
"""

from PIL import Image, ImageFilter
import numpy as np


def apply_grayscale(img):
    """
    Vectorized grayscale conversion using luminance formula.
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)

    # ITU-R BT.601
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def apply_gaussian_blur(img):
    """
    Gaussian Blur using a 3x3 smoothing kernel.
    
    Kernel:     [1, 2, 1]
                [2, 4, 2]  * (1/16)
                [1, 2, 1]
    
    The kernel is normalized by dividing by 16 (sum of weights).
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image with Gaussian blur applied
    """
    # 3x3 Gaussian kernel (approximation)
    # Center has highest weight, corners have lowest
    kernel = (
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    )
    # Scale factor of 16 normalizes the kernel (1+2+1+2+4+2+1+2+1 = 16)
    return img.filter(ImageFilter.Kernel((3, 3), kernel, scale=16))


def apply_sobel_edge_detection(img):
    """
    Sobel Edge Detection using gradient magnitude.
    
    Sobel X kernel (horizontal edges):    Sobel Y kernel (vertical edges):
        [-1, 0, 1]                            [-1, -2, -1]
        [-2, 0, 2]                            [ 0,  0,  0]
        [-1, 0, 1]                            [ 1,  2,  1]
    
    Gradient magnitude: sqrt(Gx² + Gy²)
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image with edges detected (grayscale)
    """
    # Convert to grayscale for edge detection
    gray = img.convert("L")

    # Sobel kernels for X and Y gradients
    sx = (-1, 0, 1, -2, 0, 2, -1, 0, 1)  # Horizontal edges
    sy = (-1, -2, -1, 0, 0, 0, 1, 2, 1)  # Vertical edges

    # Apply Sobel kernels
    gx = np.array(gray.filter(ImageFilter.Kernel((3, 3), sx, scale=1)), dtype=np.float32)
    gy = np.array(gray.filter(ImageFilter.Kernel((3, 3), sy, scale=1)), dtype=np.float32)

    # Compute gradient magnitude: sqrt(Gx² + Gy²)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Clip values to valid range [0, 255] instead of normalizing
    # This preserves the actual gradient strength
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return Image.fromarray(magnitude)


def apply_sharpening(img):
    """
    Image Sharpening using a Laplacian-based kernel.
    
    Kernel:     [ 0, -1,  0]
                [-1,  5, -1]
                [ 0, -1,  0]
    
    This enhances edges by subtracting a blurred version from the original.
    Center value of 5 = 1 (original) + 4 (enhancement factor)
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image with enhanced edges and details
    """
    # Sharpening kernel that enhances edges
    kernel = (
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    )
    return img.filter(ImageFilter.Kernel((3, 3), kernel, scale=1))

def apply_brightness(img, factor=1.5):
    if img.mode == "RGBA":
        rgb, alpha = img.convert("RGBA").split()[:3], img.split()[3]
    else:
        img = img.convert("RGB")
        rgb = img.split()
        alpha = None

    channels = [
        ch.point(lambda i: min(255, int(i * factor))) for ch in rgb
    ]

    out = Image.merge("RGB", channels)
    if alpha:
        out.putalpha(alpha)
    return out
