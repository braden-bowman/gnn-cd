#!/usr/bin/env python3
# Memory monitoring utilities for processing large datasets

import os
import psutil
import time
import torch
import gc
from functools import wraps

def get_memory_usage():
    """Get current memory usage for the Python process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def print_memory_usage(label=""):
    """Print current memory usage"""
    mem_usage = get_memory_usage()
    print(f"Memory usage {label}: {mem_usage:.2f} MB")
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"GPU memory: allocated={gpu_allocated:.2f} MB, reserved={gpu_reserved:.2f} MB")

def memory_monitor(func):
    """Decorator to monitor memory usage before and after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print_memory_usage(f"Before {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            raise
        finally:
            end_time = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print_memory_usage(f"After {func.__name__}")
            print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
            
        return result
    return wrapper

class MemoryTracker:
    """Context manager for tracking memory usage within a block of code"""
    def __init__(self, label=""):
        self.label = label
        
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.start_time = time.time()
        self.start_mem = get_memory_usage()
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            self.start_gpu = 0
            
        print(f"Starting {self.label}: CPU={self.start_mem:.2f} MB, GPU={self.start_gpu:.2f} MB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        end_time = time.time()
        end_mem = get_memory_usage()
        if torch.cuda.is_available():
            end_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            end_gpu = 0
            
        print(f"Finished {self.label}: CPU={end_mem:.2f} MB (Δ={end_mem-self.start_mem:.2f} MB), "
              f"GPU={end_gpu:.2f} MB (Δ={end_gpu-self.start_gpu:.2f} MB), "
              f"Time: {end_time-self.start_time:.2f}s")

def optimize_memory_usage():
    """Optimize memory usage by cleaning up caches and unused objects"""
    # Clear Python garbage collector
    gc.collect()
    
    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Suggest to run Python process with limited memory growth
    if 'OMP_NUM_THREADS' not in os.environ:
        print("Tip: Set OMP_NUM_THREADS environment variable to limit memory usage")
        print("Example: export OMP_NUM_THREADS=4")
    
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ and torch.cuda.is_available():
        print("Tip: Configure PyTorch memory allocator")
        print("Example: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    
    return get_memory_usage()

# Example usage
if __name__ == "__main__":
    print("Memory monitoring utilities")
    print_memory_usage("at start")
    
    # Example with decorator
    @memory_monitor
    def memory_intensive_function():
        """Example function that uses memory"""
        # Create a large tensor
        large_tensor = torch.randn(1000, 1000)
        time.sleep(1)  # Simulate work
        return large_tensor.shape
    
    result = memory_intensive_function()
    print(f"Result: {result}")
    
    # Example with context manager
    with MemoryTracker("test block"):
        # Create a large list
        large_list = [i for i in range(1000000)]
        time.sleep(1)  # Simulate work
    
    # Optimize memory
    freed_memory = optimize_memory_usage()
    print(f"Memory after optimization: {freed_memory:.2f} MB")