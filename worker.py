"""
worker.py - Image Processing Worker Function

This module contains the worker function that processes individual images.
Each worker process receives a task tuple and applies the specified filters.
The function is designed to be called by multiprocessing.Pool.map() or
concurrent.futures.ProcessPoolExecutor.map().

The worker function is the unit of parallel work - each process executes
this function independently on different images.

Author: CST435 Assignment 2
"""

import os
from PIL import Image
import filters


def process_image_task(task_data):
    """
    Worker function that processes a single image with specified filters.
    
    This function is executed by worker processes in parallel. Each process
    handles one image at a time, applying all requested filters and saving
    the results to disk.
    
    Args:
        task_data: Tuple containing (input_path, output_path, filter_mode)
            - input_path: Path to the source image
            - output_path: Base path for output (filter name will be appended)
            - filter_mode: 'all' or specific filter name
            
    Returns:
        int: 1 for success, 0 for failure
    """
    # Unpack task data
    input_path, output_path, filter_mode = task_data
    
    # Determine which filters to apply
    if filter_mode == 'all':
        # Apply all 5 filters to each image
        operations = ['grayscale', 'blur', 'sobel', 'sharpen', 'brightness']
    else:
        # Apply only the specified filter
        operations = [filter_mode]

    try:
        # Load the source image once (efficient for multiple filter operations)
        original_img = Image.open(input_path).convert("RGB")
        
        # Parse output path for filename construction
        file_dir, file_name = os.path.split(output_path)
        name_root, ext = os.path.splitext(file_name)
        
        # Create output directory if it doesn't exist
        os.makedirs(file_dir, exist_ok=True)

        # Apply each filter and save the result
        for op in operations:
            # Select and apply the appropriate filter function
            if op == 'grayscale':
                result_img = filters.apply_grayscale(original_img)
            elif op == 'blur':
                result_img = filters.apply_gaussian_blur(original_img)
            elif op == 'sobel':
                result_img = filters.apply_sobel_edge_detection(original_img)
            elif op == 'sharpen':
                result_img = filters.apply_sharpening(original_img)
            elif op == 'brightness':
                result_img = filters.apply_brightness(original_img)
            else:
                # Unknown filter - keep original image
                result_img = original_img

            # Construct output filename with filter suffix
            # Example: "apple_pie_12345_sobel.jpg"
            new_filename = f"{name_root}_{op}{ext}"
            final_path = os.path.join(file_dir, new_filename)
            
            # Save the processed image
            result_img.save(final_path)
            
        return 1  # Success indicator
        
    except Exception as e:
        # Log errors but don't crash the worker process
        print(f"Error processing {input_path}: {e}")
        return 0  # Failure indicator