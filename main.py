"""
main.py - Parallel Image Processing Benchmark

This module is the main entry point for the parallel image processing system.
It benchmarks three Python parallel computing paradigms:
1. multiprocessing.Pool - Low-level process pool implementation
2. concurrent.futures.ProcessPoolExecutor - High-level executor-based parallelism (processes)
3. concurrent.futures.ThreadPoolExecutor - High-level executor-based parallelism (threads)

The program processes images with configurable filters and measures:
- Execution time
- Speedup (compared to single-core baseline)
- Efficiency (speedup / number of cores)
- Throughput (images processed per second)

Usage:
    python main.py

Author: CST435 Assignment 2
"""

import time
import os
import multiprocessing
import concurrent.futures
import json
import utils
import worker

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to the Food-101 dataset (downloaded from Kaggle)
SOURCE_DIR = "C:/Users/User/.cache/kagglehub/datasets/dansbecker/food-101/versions/1" # Change path

# Directory where processed images will be saved
OUTPUT_DIR = "processed_images"

# Output file for performance metrics
JSON_FILENAME = "performance_metrics.json"

# Number of images to process (create manageable subset)
MAX_IMAGES = 4000

# Filter mode: 'grayscale', 'blur', 'sobel', 'sharpen', 'brightness', or 'all'
CHOSEN_FILTER = 'all'


# =============================================================================
# PARALLEL PARADIGM 1: multiprocessing.Pool
# =============================================================================
def run_multiprocessing(tasks, num_cores):
    """
    Execute image processing using multiprocessing.Pool.
    
    This paradigm uses Python's multiprocessing module to create a pool
    of worker processes. The Pool.map() function automatically distributes
    tasks across workers and handles inter-process communication.
    
    Args:
        tasks: List of task tuples (input_path, output_path, filter_mode)
        num_cores: Number of worker processes to spawn
        
    Returns:
        float: Execution time in seconds
    """
    print(f"   [Multiprocessing] Cores: {num_cores} | Filter: {CHOSEN_FILTER}")
    start = time.time()
    
    # Create a pool of worker processes
    # Context manager ensures proper cleanup of processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # map() blocks until all tasks complete
        pool.map(worker.process_image_task, tasks)
        
    duration = time.time() - start
    print(f"   -> Time: {duration:.4f}s")
    return duration


# =============================================================================
# PARALLEL PARADIGM 2: concurrent.futures.ProcessPoolExecutor
# =============================================================================
def run_futures(tasks, num_cores):
    """
    Execute image processing using concurrent.futures.ProcessPoolExecutor.
    
    This paradigm provides a high-level interface for process-based parallelism.
    It offers a cleaner API with Future objects for async result handling,
    while still using separate processes to bypass the GIL.
    
    Args:
        tasks: List of task tuples (input_path, output_path, filter_mode)
        num_cores: Number of worker processes (max_workers)
        
    Returns:
        float: Execution time in seconds
    """
    print(f"   [ProcessPoolExecutor] Cores: {num_cores} | Filter: {CHOSEN_FILTER}")
    start = time.time()
    
    # ProcessPoolExecutor uses separate processes (not threads)
    # This is important for CPU-bound tasks like image processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # executor.map() returns an iterator; list() forces completion
        list(executor.map(worker.process_image_task, tasks))
        
    duration = time.time() - start
    print(f"   -> Time: {duration:.4f}s")
    return duration


# =============================================================================
# PARALLEL PARADIGM 3: concurrent.futures.ThreadPoolExecutor
# =============================================================================
def run_threading(tasks, num_cores):
    """
    Execute image processing using concurrent.futures.ThreadPoolExecutor.
    
    This paradigm uses threads instead of processes. Due to Python's GIL
    (Global Interpreter Lock), threads cannot achieve true parallelism for
    CPU-bound tasks. However, this implementation is included for comparison
    to demonstrate the GIL's impact on performance.
    
    Args:
        tasks: List of task tuples (input_path, output_path, filter_mode)
        num_cores: Number of worker threads (max_workers)
        
    Returns:
        float: Execution time in seconds
    """
    print(f"   [ThreadPoolExecutor] Threads: {num_cores} | Filter: {CHOSEN_FILTER}")
    start = time.time()
    
    # ThreadPoolExecutor uses threads (subject to GIL)
    # For CPU-bound tasks, this will NOT achieve true parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # executor.map() returns an iterator; list() forces completion
        list(executor.map(worker.process_image_task, tasks))
        
    duration = time.time() - start
    print(f"   -> Time: {duration:.4f}s")
    return duration


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Get List of Images from the dataset
    raw_tasks = utils.get_image_tasks(SOURCE_DIR, OUTPUT_DIR, MAX_IMAGES)
    
    if not raw_tasks:
        print("No images found.")
    else:
        # Add filter mode to each task tuple
        tasks_with_filter = [t + (CHOSEN_FILTER,) for t in raw_tasks]
        
        # 2. Define Core Configurations to Test
        # We MUST include 1 core to calculate Speedup/Efficiency
        max_cores = os.cpu_count() or 4
        core_configs = [1, 2, 4, 8]#, max_cores]
        unique_cores = sorted(list(set(core_configs)))
        
        print(f"Processing {len(raw_tasks)} images with ALL 5 filters.")
        print(f"Testing Core Counts: {unique_cores}\n")
        
        # Intermediate storage for raw execution times
        raw_results = {}

        # 3. Execution Loop - Run both paradigms for each core count
        for cores in unique_cores:
            print(f"--- TESTING {cores} CORE(S) ---")
            
            # Run Paradigm 1: multiprocessing
            mp_time = run_multiprocessing(tasks_with_filter, cores)
            
            # Run Paradigm 2: concurrent.futures ProcessPoolExecutor
            cf_time = run_futures(tasks_with_filter, cores)
            
            # Run Paradigm 3: concurrent.futures ThreadPoolExecutor
            th_time = run_threading(tasks_with_filter, cores)
            
            raw_results[cores] = {'mp': mp_time, 'cf': cf_time, 'th': th_time}

        # 4. Calculate Performance Metrics and Build JSON Output
        json_output = {
            "metadata": {
                "total_images": len(raw_tasks),
                "filter_mode": CHOSEN_FILTER,
                "max_physical_cores": max_cores
            },
            "results": {}
        }

        # Get baseline times (single-core execution)
        base_time_mp = raw_results[1]['mp']
        base_time_cf = raw_results[1]['cf']
        base_time_th = raw_results[1]['th']

        # Print performance report header
        print("\n" + "="*100)
        print(f"{'PERFORMANCE ANALYSIS REPORT':^100}")
        print("="*100)
        print(f"{'Cores':<6} | {'Paradigm':<20} | {'Time (s)':<10} | {'Speedup':<8} | {'Efficiency':<10} | {'Throughput (img/s)':<18}")
        print("-" * 100)

        for cores in unique_cores:
            # Prepare dictionary for this core count
            json_output["results"][cores] = {}

            # --- Multiprocessing Metrics ---
            t_mp = raw_results[cores]['mp']
            speedup_mp = base_time_mp / t_mp  # Speedup = T1 / Tn
            eff_mp = (speedup_mp / cores) * 100  # Efficiency = Speedup / n
            throughput_mp = len(raw_tasks) / t_mp  # Throughput = images / time
            
            # Save to JSON structure
            json_output["results"][cores]["multiprocessing"] = {
                "time_seconds": t_mp,
                "speedup": speedup_mp,
                "efficiency_percent": eff_mp,
                "throughput_imgs_per_sec": throughput_mp
            }
            
            print(f"{cores:<6} | {'Multiprocessing.Pool':<20} | {t_mp:<10.4f} | {speedup_mp:<8.2f} | {eff_mp:<10.1f}% | {throughput_mp:<18.2f}")

            # --- ProcessPoolExecutor Metrics ---
            t_cf = raw_results[cores]['cf']
            speedup_cf = base_time_cf / t_cf
            eff_cf = (speedup_cf / cores) * 100
            throughput_cf = len(raw_tasks) / t_cf

            # Save to JSON structure
            json_output["results"][cores]["process_pool_executor"] = {
                "time_seconds": t_cf,
                "speedup": speedup_cf,
                "efficiency_percent": eff_cf,
                "throughput_imgs_per_sec": throughput_cf
            }
            
            print(f"{cores:<6} | {'ProcessPoolExecutor':<20} | {t_cf:<10.4f} | {speedup_cf:<8.2f} | {eff_cf:<10.1f}% | {throughput_cf:<18.2f}")

            # --- ThreadPoolExecutor Metrics ---
            t_th = raw_results[cores]['th']
            speedup_th = base_time_th / t_th
            eff_th = (speedup_th / cores) * 100
            throughput_th = len(raw_tasks) / t_th

            # Save to JSON structure
            json_output["results"][cores]["thread_pool_executor"] = {
                "time_seconds": t_th,
                "speedup": speedup_th,
                "efficiency_percent": eff_th,
                "throughput_imgs_per_sec": throughput_th
            }
            
            print(f"{cores:<6} | {'ThreadPoolExecutor':<20} | {t_th:<10.4f} | {speedup_th:<8.2f} | {eff_th:<10.1f}% | {throughput_th:<18.2f}")
            print("-" * 100)

        # 5. Save Full Data to JSON File
        try:
            with open(JSON_FILENAME, 'w') as f:
                json.dump(json_output, f, indent=4)
            print(f"\n[Success] Full performance metrics saved to '{JSON_FILENAME}'")
        except IOError as e:
            print(f"\n[Error] Failed to save metrics: {e}")
        
        print("Done. Copy the table above for your report.")
