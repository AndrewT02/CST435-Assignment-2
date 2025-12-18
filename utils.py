"""
utils.py - Utility Functions

This module provides utility functions for the parallel image processing system.
Currently includes functions for:
- Image discovery and task generation

Author: CST435 Assignment 2
"""

import os


def get_image_tasks(source_dir, output_base_dir, max_images=20):
    """
    Scan a directory for images and generate processing tasks.
    
    This function walks through the source directory recursively,
    finds all image files, and creates task tuples that map input
    paths to output paths while preserving directory structure.
    
    Args:
        source_dir: Root directory containing source images
        output_base_dir: Root directory for processed output images
        max_images: Maximum number of images to process (default: 20)
        
    Returns:
        List of tuples: [(input_path, output_path), ...]
        
    Supported formats:
        .jpg, .jpeg, .png
    """
    tasks = []
    count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check for supported image extensions
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Full path to source image
                input_path = os.path.join(root, file)
                
                # Preserve directory structure in output
                # e.g., source/apple_pie/123.jpg -> output/apple_pie/123.jpg
                rel_path = os.path.relpath(input_path, source_dir)
                output_path = os.path.join(output_base_dir, rel_path)
                
                # Add task tuple to list
                tasks.append((input_path, output_path))
                
                count += 1
                # Stop when we've collected enough images
                if count >= max_images:
                    return tasks
                    
    return tasks